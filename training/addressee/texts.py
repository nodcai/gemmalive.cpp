"""Building blocks for synthetic addressee data: speech meant for the
assistant, and speech that is not (side talk, backchannels).

Written for this project; combined at random by make_data.py.
"""

# People someone in the room might talk to instead of the assistant.
VOCATIVES = [
    "Mom", "Dad", "Honey", "Babe", "Sweetie", "Grandma", "Grandpa", "Guys", "Kids", "Buddy",
    "Sarah", "Tom", "Alex", "Maria", "Jake", "Priya", "Chen", "Olivia", "Marcus", "Emma",
    "Noah", "Lena", "Omar", "Sophie", "Daniel", "Aisha", "Ben", "Kate", "Luis", "Mia",
    "Doctor Lee", "Coach", "Officer", "Sir", "Ma'am", "Miss Patel", "Mister Brown", "Professor",
    "Hey man", "Dude", "Bro", "Sis", "Auntie", "Uncle Joe", "Boss", "Team", "Everyone", "Folks",
    "Waiter", "Excuse me", "Hey you", "Love", "Darling", "Pal", "Neighbor", "Nurse", "Captain",
]

# Remarks to someone else in the room.
SIDE_REMARKS = [
    "can you grab the door?", "dinner's ready.", "the microwave just stopped working.",
    "where did you put the car keys?", "turn the TV down a bit, please.", "I'll be there in a minute.",
    "did you feed the dog?", "your phone is ringing.", "we're out of milk again.",
    "can you pass me the salt?", "don't forget your jacket.", "the kids are asleep, keep it down.",
    "have you seen my glasses?", "I'm almost done here.", "your laptop is still on the table.",
    "these shoes are killing me.", "can you take out the trash?", "the meeting got moved to three.",
    "I left the charger in your car.", "close the window, it's freezing.", "who finished the coffee?",
    "the package arrived this morning.", "my ankle still hurts a little.", "is this the right form?",
    "you're blocking the TV.", "hand me that towel, would you?", "I think the oven is still on.",
    "we need to leave in ten minutes.", "your mom called earlier.", "did you lock the back door?",
    "can you watch the stove for a sec?", "the Wi-Fi is down again.", "look at this picture.",
    "I'm going to the store, need anything?", "that was the best pizza ever.", "stop kicking my chair.",
    "the plants need water.", "it's your turn to do the dishes.", "I can't find the remote.",
    "the bus is late again.", "you left the lights on.", "grab me a soda from the fridge.",
    "we should paint this room blue.", "your shoelace is untied.", "is it raining outside?",
    "put your shoes away, please.", "I signed the papers already.", "the printer is jammed.",
    "let's order Thai tonight.", "the baby needs changing.", "your appointment is at four.",
    "be quiet, I'm thinking.", "how was school today?", "I'm so tired today.",
    "did anyone see my wallet?", "the cat knocked over the plant.", "we're running late.",
    "save me a seat.", "can I borrow your pen?", "the game starts at eight.",
    "you look great in that shirt.", "I parked on the second floor.", "check the mailbox, please.",
    "the dishwasher is making that noise again.", "your coffee's getting cold.", "wait for me outside.",
    "who's picking up grandma?", "this traffic is unbelievable.", "I already paid for lunch.",
    "the fridge door is open.", "your sister is on the phone.", "did you finish your homework?",
    "we have guests coming at seven.", "that song is stuck in my head.", "watch your step there.",
    "move your bag off the chair.", "the light bulb in the hall is out.", "I told you so.",
    "keep an eye on the soup.", "your keys are by the door.", "the neighbors are so loud tonight.",
]

# Talking to someone on the phone or a person who just walked in.
OTHER_PARTY = [
    "hey, can I call you back in five?", "yeah, I'm on my way now.", "no, Jen, the other one, the red one.",
    "sorry, I'm in the middle of something.", "okay, see you at six then.", "tell her I said hi.",
    "hold on, someone's at the door.", "yes, the order number is on the receipt.",
    "I'll send you the address in a text.", "no worries, take your time.", "it's on the left after the bridge.",
    "can you speak up? I can barely hear you.", "great, thanks for letting me know.",
    "I'll be right there, give me a second.", "just leave it on the porch, thanks.",
    "sorry, wrong number.", "yeah, put it on my tab.", "can we do Thursday instead?",
    "I'm not sure, let me check with my wife.", "uh, two coffees and a muffin, please.",
]

# Listener feedback that does not take the floor.
BACKCHANNELS = [
    "mm-hm", "uh-huh", "yeah", "yeah yeah", "right", "okay", "ok", "sure", "I see", "got it",
    "mhm", "yep", "oh", "oh wow", "wow", "nice", "cool", "really", "interesting", "hmm",
    "uh huh", "right right", "oh okay", "ah", "ahh I see", "makes sense", "true", "totally",
    "exactly", "yes", "oh nice", "huh", "okay okay", "alright", "sounds good", "I know",
    "oh really", "no way", "ha", "haha",
]

# Speech meant for the assistant: taking the floor while it talks.
INTERRUPTIONS = [
    "wait, stop.", "hold on a second.", "sorry, can you repeat that?", "no, that's not what I meant.",
    "actually, never mind.", "stop, stop.", "wait, what did you say?", "can you say that again?",
    "okay, that's enough, thanks.", "no, I asked about tomorrow.", "hang on, go back a bit.",
    "sorry to interrupt, but", "actually, let me ask something else.", "slower, please.",
    "can you speak a bit slower?", "that's wrong.", "no no, the other one.", "skip that part.",
    "just give me the short version.", "okay, but what about the weekend?", "wait, how much was that?",
    "hmm, can you explain that again?", "pause for a second.", "never mind, I got it.",
    "that's not right, check again.", "and what about the price?", "okay, thanks, that's all.",
    "no, I mean in Celsius.", "sorry, I didn't catch the last part.", "can you make it shorter?",
    "wait, before you go on", "what does that mean?", "say that one more time.",
    "no, cancel that.", "actually, change it to six.", "hold on, let me write this down.",
]

# Requests to an assistant, as an addition to questions from Dolly.
COMMANDS = [
    "set a timer for ten minutes.", "what's the weather like tomorrow?", "remind me to call the dentist.",
    "play some relaxing music.", "how do I make pancakes?", "what time is it in Tokyo?",
    "turn off the living room lights.", "add eggs to my shopping list.", "tell me a joke.",
    "how far is the moon?", "what's on my calendar today?", "translate thank you into Spanish.",
    "how many ounces are in a cup?", "what's the capital of Australia?", "read me the news.",
    "set an alarm for seven a.m.", "how long should I boil an egg?", "what's fifteen percent of eighty?",
    "who wrote Pride and Prejudice?", "find a recipe for lasagna.", "what's a good name for a dog?",
    "how do I fix a flat tire?", "what's the population of Canada?", "give me a quick workout.",
    "explain how vaccines work.", "what should I cook tonight?", "is it going to rain later?",
    "how do you spell necessary?", "what's the fastest way to the airport?", "summarize the plot of Hamlet.",
]

# Discourse markers that open a new topic or a correction. On both sides:
# with a question to the assistant, or with side talk to someone named.
MARKERS = [
    "By the way,", "Actually,", "Hold on,", "Wait,", "Oh,", "Oh, before I forget,", "Before I forget,",
    "Quick question,", "Sorry,", "Also,", "One more thing,", "Hmm,", "Hey,", "Okay so,", "Real quick,",
    "Oh, and", "Just wondering,", "Speaking of which,", "Hang on,", "Oh wait,", "Random question,",
    "Anyway,", "Oh right,", "Changing topics,", "So,", "Listen,", "Oh, also,", "On another note,",
    "While I remember,", "Oh, one thing,", "Sorry to interrupt,", "Right,", "Okay, different question,",
]

# Things a user might tell the assistant.
STATEMENTS = [
    "I signed up for a cooking class.", "I just got a new puppy.", "I'm thinking about learning guitar.",
    "I finally finished that book.", "my flight got delayed again.", "I started running in the mornings.",
    "I need to plan a birthday party.", "I've been sleeping badly this week.", "I want to eat healthier.",
    "I'm moving to a new apartment next month.", "my car is making a weird noise.", "I got the job!",
    "I have a dentist appointment tomorrow.", "I'm trying to save for a vacation.", "I lost my keys again.",
    "I'm hosting dinner on Saturday.", "my laptop keeps freezing.", "I'm learning Spanish.",
    "I forgot to call my mom.", "I want to start a garden.", "I'm feeling a bit stressed today.",
    "I booked a trip to Lisbon.", "I need a gift idea for my brother.", "I ran my first ten k.",
]

# Questions and requests to an assistant, written for this project. They
# replace Dolly-15k, whose share-alike licence would carry over unclearly.
QUESTIONS = [
    # weather and time
    "what's the weather going to be like this weekend?", "do I need an umbrella today?",
    "how cold will it get tonight?", "when is sunset today?", "what time is it in London right now?",
    "is it going to snow tomorrow?", "what's the humidity like outside?", "how windy is it at the beach today?",
    "what day of the week is Christmas this year?", "how many days until my birthday on June third?",
    "when does daylight saving time start?", "what's the UV index this afternoon?",
    # cooking and food
    "how long do I bake chicken breasts at two hundred degrees?", "what can I make with eggs, spinach and cheese?",
    "how do I make a simple tomato sauce?", "can I freeze cooked rice?", "what's a good substitute for buttermilk?",
    "how many grams are in a cup of flour?", "how long does it take to hard boil an egg?",
    "give me a quick vegetarian dinner idea.", "how do I know if an avocado is ripe?", "what wine goes well with salmon?",
    "how do I keep basil fresh?", "what's the difference between baking soda and baking powder?",
    "how much water do I need for two cups of rice?", "suggest a dessert I can make in twenty minutes.",
    "how do I make cold brew coffee?", "is it safe to eat raw cookie dough?", "how long can I keep leftovers in the fridge?",
    "what's an easy marinade for tofu?",
    # health and fitness
    "how many steps should I walk a day?", "what are some good stretches for lower back pain?",
    "how much water should I drink in a day?", "what's a healthy breakfast that keeps me full?",
    "how can I fall asleep faster?", "give me a ten minute workout I can do at home.",
    "what are the symptoms of dehydration?", "how many calories are in a banana?", "is it okay to run every day?",
    "how do I lower my resting heart rate?", "what helps with a sore throat?", "how long should a power nap be?",
    "what's a good way to start meditating?", "how do I improve my posture at my desk?",
    # travel and navigation
    "how long does it take to fly from Berlin to New York?", "what should I pack for a weekend in Rome?",
    "do I need a visa to visit Japan?", "what's the best time of year to visit Iceland?",
    "how far is the nearest gas station?", "what's the traffic like on my way to work?",
    "find me a pharmacy that's open late.", "how do I get to the train station from here?",
    "what are some must see places in Lisbon?", "how much should I tip in a restaurant in the US?",
    "what's the exchange rate from euros to dollars?", "how early should I get to the airport for an international flight?",
    "can I bring a power bank on a plane?", "what's a good day trip from Barcelona?",
    # technology help
    "how do I take a screenshot on my laptop?", "why is my phone battery draining so fast?",
    "how do I reset my router?", "how can I free up storage on my phone?", "what's the difference between RAM and storage?",
    "how do I connect my headphones with bluetooth?", "how do I make a strong password?",
    "why is my Wi-Fi so slow in the evening?", "how do I back up my photos?", "what does a VPN do?",
    "how do I turn on dark mode?", "how can I tell if an email is a scam?", "how do I update my laptop?",
    "what's a good free photo editor?",
    # math and units
    "what's twenty percent of eighty five?", "how many miles is ten kilometers?", "what's twelve times seventeen?",
    "convert seventy degrees Fahrenheit to Celsius.", "how many seconds are in a day?",
    "what's the square root of one hundred forty four?", "split a bill of ninety euros between four people.",
    "how many liters are in a gallon?", "what's seven eighths as a decimal?", "how many feet are in a meter?",
    "if I save fifty dollars a week, how much do I have after a year?", "what's the area of a circle with radius three?",
    # facts and trivia
    "who painted the Mona Lisa?", "how tall is Mount Everest?", "what's the longest river in the world?",
    "when did the first person land on the moon?", "how many bones are in the human body?",
    "what's the largest ocean on Earth?", "who invented the telephone?", "how far away is the sun?",
    "what language do they speak in Brazil?", "why is the sky blue?", "how do airplanes stay in the air?",
    "what's the smallest country in the world?", "how many players are on a soccer team?",
    "who wrote the Odyssey?", "what's the capital of Canada?", "how old is the universe?",
    "why do cats purr?", "how do bees make honey?", "what causes thunder?", "how many hearts does an octopus have?",
    # language and writing
    "how do you say good morning in Italian?", "what's another word for happy?", "how do I spell restaurant?",
    "what does the word ephemeral mean?", "help me write a short thank you note to my neighbor.",
    "is it affect or effect in this sentence?", "translate where is the train station into French.",
    "give me a rhyme for orange.", "what's the plural of cactus?", "write a short birthday message for my sister.",
    "help me say no politely to a meeting.", "what's a good opening line for a cover letter?",
    # music and entertainment
    "play something upbeat for cooking.", "what's a good movie for a rainy Sunday?", "recommend a podcast about history.",
    "who sings this song?", "what are some good board games for four people?", "turn the volume down a little.",
    "skip this song.", "what's a good book for a long flight?", "recommend a funny TV series.",
    "play some quiet piano music.", "what time does the game start tonight?", "who won the last World Cup?",
    # home and errands
    "add milk and bread to my shopping list.", "set a timer for twelve minutes.", "remind me to water the plants tomorrow.",
    "turn off the lights in the kitchen.", "set the thermostat to twenty one degrees.", "what's on my calendar this afternoon?",
    "remind me to call the plumber at nine.", "how do I get a red wine stain out of a carpet?",
    "how often should I change my bed sheets?", "what's the best way to clean a cast iron pan?",
    "wake me up at six thirty.", "cancel my alarm for tomorrow.", "how do I unclog a drain without chemicals?",
    "when is the recycling picked up?", "make a note that the car needs new tires.",
    # money and work
    "how can I make a simple monthly budget?", "what's the difference between a Roth and a traditional IRA?",
    "how do I ask for a raise?", "give me tips for a job interview tomorrow.", "how do I write a good email subject line?",
    "what's compound interest?", "how much should I save for an emergency fund?", "how do I prepare for a presentation?",
    "help me plan my week.", "what's a good way to stay focused while working from home?",
    # pets and kids
    "how often should I feed a puppy?", "is chocolate dangerous for dogs?", "how do I get my cat to stop scratching the sofa?",
    "what are some fun indoor games for kids?", "how much sleep does a five year old need?",
    "give me a short bedtime story about a dragon.", "how do I teach my dog to sit?", "what's a good first pet for a child?",
    "how do I help my kid with math homework?", "suggest a science experiment for kids at home.",
    # general chat and opinions
    "tell me something interesting.", "what's a good hobby to pick up this year?", "give me a fun fact about space.",
    "what should I do this weekend?", "can you recommend a good gift for my dad?", "how do I make new friends as an adult?",
    "what's a good name for a goldfish?", "tell me a riddle.", "how can I be more productive in the morning?",
    "what's the best way to learn a new language?", "how do I stop procrastinating?", "give me a motivating quote.",
    "what are some easy plants for beginners?", "how do I start running if I've never run before?",
    "explain the stock market in simple terms.", "what's the difference between a virus and bacteria?",
    "how does a rainbow form?", "why do we dream?", "what's the healthiest way to cook vegetables?",
    "how do I choose a good watermelon?",
]

# Follow-ups that lean on the answer before them: short, and often with a
# pronoun or an "and" instead of what they are about.
FOLLOW_UPS = [
    "why?", "how come?", "really?", "are you sure?", "since when?", "how so?", "like what?", "such as?",
    "what do you mean?", "can you explain that?", "can you say that again?", "what was that?",
    "tell me more.", "go on.", "and then?", "and after that?", "what happened next?", "anything else?",
    "what else?", "is that all?", "how many?", "how much?", "how long?", "how far?", "how old?", "how big is it?",
    "how long is it?", "how deep is it?", "how old is he?", "how old is she?", "how tall was he?",
    "when was that?", "when did that happen?", "where is that?", "where is it?", "who was that?", "which one?",
    "which one is better?", "which is closer?", "what about tomorrow?", "and tomorrow?", "and next week?",
    "and on Sunday?", "and in the evening?", "what about Paris?", "and in London?", "and for two people?",
    "what about the other one?", "and the second one?", "and the last one?", "and her?", "and his brother?",
    "what did she write?", "what else did she do?", "where did she grow up?", "where did they live?",
    "when did he die?", "is he still alive?", "was she married?", "did he win?", "who won?", "who else?",
    "is it far?", "is it safe?", "is it expensive?", "is it open now?", "is it worth it?", "is that a lot?",
    "is that normal?", "is that true?", "does it hurt?", "does it work?", "does that help?", "will it rain?",
    "can I do that at home?", "can you do it faster?", "can I use butter instead?", "can kids do it too?",
    "do I need a jacket then?", "should I go now?", "should I call them?", "should I worry?",
    "what should I bring?", "what should I do then?", "what would you pick?", "what would you do?",
    "how do I do that?", "how does it work?", "how does that happen?", "how did they build it?",
    "how did it end?", "how is it made?", "how do you know?", "where does it come from?",
    "what's it made of?", "what's it called?", "what does it look like?", "what does that mean?",
    "what's the difference?", "which is healthier?", "which is faster?", "what about in winter?",
    "and at night?", "and in Celsius?", "and in miles?", "and in dollars?", "and per person?",
    "and if it rains?", "and without sugar?", "and the population?", "and the second largest?",
    "and the oldest?", "and the smallest?", "and who came second?", "and before that?",
    "so what's the answer?", "so is it yes or no?", "so should I?", "so how long then?",
    "okay, and how much does it cost?", "right, and where is it?", "hmm, and why is that?",
    "oh, and when does it close?", "wait, how old was she?", "okay, but is it safe?",
    "tell me about her.", "tell me about his books.", "tell me another one.", "give me another example.",
    "one more, please.", "say it slower.", "in simple words, please.", "explain it to a child.",
    "make it shorter.", "translate that into Spanish.", "spell it.", "repeat the last part.",
]

# Ways of ending a conversation, all finished.
CLOSINGS = [
    "okay, thanks.", "thank you.", "thanks a lot.", "great, thanks.", "perfect, thank you.", "got it, thanks.",
    "alright, that's it.", "that's all.", "that's all I needed.", "okay, I'm done.", "cool, thanks.",
    "nice, thank you.", "awesome, thanks.", "okay, good to know.", "alright, thank you.", "okay, bye.",
    "thanks, bye.", "that's it for now.", "good, that helps.", "okay, that's helpful.", "thanks, that's enough.",
    "great, that's all.", "okay, perfect.", "noted, thanks.", "okay, see you later.", "all good, thanks.",
]
