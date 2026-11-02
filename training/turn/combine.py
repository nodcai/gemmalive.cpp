"""Puts a model, its audio projector and its heads into one GGUF file.

    python combine.py models/gemma-4-E2B-live-XS.gguf \
        --model models/gemma-4-E2B-it-XS-TS-Q3_K_S.gguf \
        --mmproj models/mmproj-gemma-4-E2B-it-Q8_0.gguf

The heads are the model's, <model>-heads.gguf, unless --heads names others;
they have to be trained on this model and projector. gemmalive takes the
file as --model, and needs no other for the model.

The file is the model's, with the projector's clip.* and the heads' heads.*
metadata and the tensors of both after the model's own.
"""

import argparse
import hashlib
import os

import gguf

ARCHITECTURE = "general.architecture"


def fingerprint(path):
    """SHA-256 of a GGUF file's header, as export.py records it."""
    offset = gguf.GGUFReader(path).data_offset
    with open(path, "rb") as f:
        return hashlib.sha256(f.read(offset)).hexdigest()


def value(reader, key):
    field = reader.fields.get(key)
    return field.contents() if field else None


def add_fields(w, reader, prefix, replace):
    """Copies the metadata under `prefix`, less the keys `replace` sets."""
    for field in reader.fields.values():
        if field.name.startswith("GGUF."):  # the reader's own: version and counts
            continue
        if not field.name.startswith(prefix) or field.name == ARCHITECTURE or field.name in replace:
            continue
        kind = field.types[0]
        sub = field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(field.name, field.contents(), kind, sub_type=sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--heads", default=None, help="default: <model>-heads.gguf")
    args = ap.parse_args()
    heads_path = args.heads or os.path.splitext(args.model)[0] + "-heads.gguf"

    model, mmproj, heads = (gguf.GGUFReader(p) for p in (args.model, args.mmproj, heads_path))
    for what, path in (("model", args.model), ("mmproj", args.mmproj)):
        if value(heads, f"heads.{what}.fingerprint") != fingerprint(path):
            raise SystemExit(f"{heads_path} was trained for a different {what} than {path}")
    if len({r.alignment for r in (model, mmproj, heads)}) != 1:
        raise SystemExit("the files align their tensors differently")

    name = os.path.splitext(os.path.basename(args.output))[0]
    ours = {
        "general.name": name,
        "general.description": f"{value(model, 'general.name')} with its audio projector and gemmalive's "
                               "end-of-turn and addressee heads, for gemmalive.cpp.",
    }

    w = gguf.GGUFWriter(args.output, value(model, ARCHITECTURE))
    w.data_alignment = model.alignment
    for key, text in ours.items():
        w.add_string(key, text)
    add_fields(w, model, "", ours)
    add_fields(w, mmproj, "clip.", ours)
    add_fields(w, heads, "heads.", ours)

    tensors = [t for r in (model, mmproj, heads) for t in r.tensors]
    names = [t.name for t in tensors]
    if len(set(names)) != len(names):
        raise SystemExit("the files have tensors of the same name")
    for t in tensors:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in tensors:
        w.write_tensor_data(t.data)
    w.close()
    print(f"wrote {args.output}: {len(tensors)} tensors, {os.path.getsize(args.output) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
