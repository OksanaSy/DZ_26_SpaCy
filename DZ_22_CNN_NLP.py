import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, Dropout, Flatten

df = pd.read_csv('IMDB Dataset.csv')

reviews = df['review'].values
labels = df['sentiment'].values

labels = np.array([1 if label == 'positive' else 0 for label in labels])

X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_len = 200 
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

vocab_size = 5000 
embedding_dim = 100  
num_filters = 128  
kernel_size = 5    
dropout_rate = 0.5  

model = Sequential()

model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))

model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu'))

model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())
model.add(Dropout(dropout_rate))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_data=(X_test_pad, y_test), verbose=1)

test_loss, test_acc = model.evaluate(X_test_pad, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

predictions = model.predict(X_test_pad)
predicted_labels = (predictions > 0.5).astype(int).flatten()

num_examples = 10  
for i in range(num_examples):
    print(f"Review: {X_test[i]}")
    print(f"True Label: {'positive' if y_test[i] == 1 else 'negative'}")
    print(f"Predicted Label: {'positive' if predicted_labels[i] == 1 else 'negative'}")
    print("-" * 50)

"""run results:
Epoch 1/5
625/625 ━━━━━━━━━━━━━━━━━━━━ 18s 27ms/step - accuracy: 0.7201 - loss: 0.4913 - val_accuracy: 0.8978 - val_loss: 0.2538
Epoch 2/5
625/625 ━━━━━━━━━━━━━━━━━━━━ 18s 28ms/step - accuracy: 0.9147 - loss: 0.2139 - val_accuracy: 0.8970 - val_loss: 0.2545
Epoch 3/5
625/625 ━━━━━━━━━━━━━━━━━━━━ 17s 27ms/step - accuracy: 0.9394 - loss: 0.1628 - val_accuracy: 0.8943 - val_loss: 0.2610
Epoch 4/5
625/625 ━━━━━━━━━━━━━━━━━━━━ 17s 28ms/step - accuracy: 0.9613 - loss: 0.1099 - val_accuracy: 0.8936 - val_loss: 0.3024
Epoch 5/5
625/625 ━━━━━━━━━━━━━━━━━━━━ 17s 27ms/step - accuracy: 0.9793 - loss: 0.0645 - val_accuracy: 0.8926 - val_loss: 0.3466
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.8906 - loss: 0.3460
Test Loss: 0.3466
Test Accuracy: 0.8926
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step
Review: I really liked this Summerslam due to the look of the arena, the curtains and just the look overall was interesting to me for some reason. Anyways, this could have been one of the best Summerslam's ever if the WWF didn't have Lex Luger in the main event against Yokozuna, now for it's time it was ok to have a huge fat man vs a strong man but I'm glad times have changed. It was a terrible main event just like every match Luger is in is terrible. Other matches on the card were Razor Ramon vs Ted Dibiase, Steiner Brothers vs Heavenly Bodies, Shawn Michaels vs Curt Hening, this was the event where Shawn named his big monster of a body guard Diesel, IRS vs 1-2-3 Kid, Bret Hart first takes on Doink then takes on Jerry Lawler and stuff with the Harts and Lawler was always very interesting, then Ludvig Borga destroyed Marty Jannetty, Undertaker took on Giant Gonzalez in another terrible match, The Smoking Gunns and Tatanka took on Bam Bam Bigelow and the Headshrinkers, and Yokozuna defended the world title against Lex Luger this match was boring and it has a terrible ending. However it deserves 8/10
True Label: positive
Predicted Label: positive
--------------------------------------------------
Review: Not many television shows appeal to quite as many different kinds of fans like Farscape does...I know youngsters and 30/40+ years old;fans both Male and Female in as many different countries as you can think of that just adore this T.V miniseries. It has elements that can be found in almost every other show on T.V, character driven drama that could be from an Australian soap opera; yet in the same episode it has science fact & fiction that would give even the hardiest "Trekkie" a run for his money in the brainbender stakes! Wormhole theory, Time Travel in true equational form...Magnificent. It embraces cultures from all over the map as the possibilities are endless having multiple stars and therefore thousands of planets to choose from.<br /><br />With such a broad scope; it would be expected that nothing would be able to keep up the illusion for long, but here is where "Farscape" really comes into it's own element...It succeeds where all others have failed, especially the likes of Star Trek (a universe with practically zero Kaos element!) They ran out of ideas pretty quickly + kept rehashing them! Over the course of 4 seasons they manage to keep the audience's attention using good continuity and constant character evolution with multiple threads to every episode with unique personal touches to camera that are specific to certain character groups within the whole. This structure allows for an extremely large area of subject matter as loyalties are forged and broken in many ways on many many issues. I happened to see the pilot (Premiere) in passing and just had to keep tuning in after that to see if Crichton would ever "Get the girl", after seeing them all on television I was delighted to see them available on DVD & I have to admit that it was the only thing that kept me sane whilst I had to do a 12 hour night shift and developed chronic insomnia...Farscape was the only thing to get me through those extremely long nights...<br /><br />Do yourself a favour; Watch the pilot and see what I mean...<br /><br />Farscape Comet
True Label: positive
Predicted Label: positive
--------------------------------------------------
Review: The film quickly gets to a major chase scene with ever increasing destruction. The first really bad thing is the guy hijacking Steven Seagal would have been beaten to pulp by Seagal's driving, but that probably would have ended the whole premise for the movie.<br /><br />It seems like they decided to make all kinds of changes in the movie plot, so just plan to enjoy the action, and do not expect a coherent plot. Turn any sense of logic you may have, it will reduce your chance of getting a headache.<br /><br />I does give me some hope that Steven Seagal is trying to move back towards the type of characters he portrayed in his more popular movies.
True Label: negative
Predicted Label: negative
--------------------------------------------------
Review: Jane Austen would definitely approve of this one!<br /><br />Gwyneth Paltrow does an awesome job capturing the attitude of Emma. She is funny without being excessively silly, yet elegant. She puts on a very convincing British accent (not being British myself, maybe I'm not the best judge, but she fooled me...she was also excellent in "Sliding Doors"...I sometimes forget she's American ~!). <br /><br />Also brilliant are Jeremy Northam and Sophie Thompson and Phyllida Law (Emma Thompson's sister and mother) as the Bates women. They nearly steal the show...and Ms. Law doesn't even have any lines!<br /><br />Highly recommended.
True Label: positive
Predicted Label: positive
--------------------------------------------------
Review: Expectations were somewhat high for me when I went to see this movie, after all I thought Steve Carell could do no wrong coming off of great movies like Anchorman, The 40 Year-Old Virgin, and Little Miss Sunshine. Boy, was I wrong.<br /><br />I'll start with what is right with this movie: at certain points Steve Carell is allowed to be Steve Carell. There are a handful of moments in the film that made me laugh, and it's due almost entirely to him being given the wiggle-room to do his thing. He's an undoubtedly talented individual, and it's a shame that he signed on to what turned out to be, in my opinion, a total train-wreck.<br /><br />With that out of the way, I'll discuss what went horrifyingly wrong.<br /><br />The film begins with Dan Burns, a widower with three girls who is being considered for a nationally syndicated advice column. He prepares his girls for a family reunion, where his extended relatives gather for some time with each other.<br /><br />The family is high atop the list of things that make this an awful movie. No family behaves like this. It's almost as if they've been transported from Pleasantville or Leave it to Beaver. They are a caricature of what we think a family is when we're 7. It reaches the point where they become obnoxious and simply frustrating. Touch football, crossword puzzle competitions, family bowling, and talent shows ARE NOT HOW ACTUAL PEOPLE BEHAVE. It's almost sickening.<br /><br />Another big flaw is the woman Carell is supposed to be falling for. Observing her in her first scene with Steve Carell is like watching a stroke victim trying to be rehabilitated. What I imagine is supposed to be unique and original in this woman comes off as mildly retarded.<br /><br />It makes me think that this movie is taking place on another planet. I left the theater wondering what I just saw. After thinking further, I don't think it was much.
True Label: negative
Predicted Label: negative
--------------------------------------------------
Review: I've watched this movie on a fairly regular basis for most of my life, and it never gets old. For all the snide remarks and insults (mostly from David Spade), "Tommy Boy" has a giant heart. And that's what keeps this movie funny after all these years.<br /><br />Tommy Callahan (Chris Farley) is the son of Big Tom Callahan (Brian Dennehy), master car parts salesman, and has ridden on that all his life. But after his died dies on his wedding day, Tommy learns that the company is in debt, and about to be bought by Ray Zalinsky (Dan Akroyd), the owner of a huge car parts company. So in order to save the company, Tommy has to go on the road to sell the company's new brake pads. Along for the ride, though not by choice, is Richard Hayden (David Spade) a former classmate of Tommy's who was Big Tom's right-hand man.<br /><br />The movie rides on the chemistry between the two SNL stars (and real-life best friends) Chris Farley and David Spade. The duo has enough comic energy going between them to power the world. It's the big, dumb guy versus the smart little guy. It works, and some of their scenes are unforgettably funny. Farley and Spade are actually decent dramatic actors as well. Although the film is primarily a comedy, it has its fair share of drama, but Spade and especially Farley are just as good there as when they're making the audience laugh.<br /><br />Forgive me, but I have to talk about Chris Farley a little more. I read his biography ("The Chris Farley Show: A Biography in Three Acts," for anyone who cares), and understanding who Chris was in real life made this movie more special to me. Chris Farley was a genuinely good person who struggled, and ultimately failed to conquer his addictions. Although this was the first movie he had a major role in, it is his best film. It really showed who he was, and just how much talent he had. Knowing Chris's story adds another layer to this movie, although it doesn't make it any less funny.<br /><br />Farley and Spade are matched with a good on screen cast. Rob Lowe is suitably slimy as Tommy's "new brother," and Bo Derek is solid as his step-mother. Brian Dennehy is great as Big Tom. Dennehy makes it easy to believe that they're father in son. Big Tom is just as crazy as his son, although he's smarter and more mature. Dan Akroyd gives one of his best performances as Zalinsky, giving Tommy the hard truth behind advertising. Julie Warner is also good as Tommy's love interest, Michelle.<br /><br />For me, Peter Segal is one of the great comedy directors. He keeps the pace quick and energetic, but most importantly, he knows how to make comedy funny. He doesn't belabor the jokes, and he understands that funny actors know what they're doing and he allows them to do it. But Segal goes a step further. He gives "Tommy Boy" a friendly, almost nostalgic tone that both tugs the heartstrings (genuinely) and tickles the funnybone.<br /><br />Critics didn't like "Tommy Boy." Shame on them. A movie doesn't have to be super sophisticated or subversively intellectual to be funny (God forbid Farley and Spade were forced to do muted comedy a la "The Office"). This is a great movie and one of my all-time favorites.
True Label: positive
Predicted Label: positive
--------------------------------------------------
Review: For once a story of hope highlighted over the tragic reality our youth face. Favela Rising draws one into a scary, unsafe and unfair world and shows through beautiful color and moving music how one man and his dedicated friends choose not to accept that world and change it through action and art. An entertaining, interesting, emotional, aesthetically beautiful film. I showed this film to numerous high school students as well who all live in neighborhoods with poverty and and gun violence and they were enamored with Anderson, the protagonist. I recommend this film to all ages over 13 (due to subtitles and some images of death) from all backgrounds.
True Label: positive
Predicted Label: positive
--------------------------------------------------
Review: Okay, I didn't get the Purgatory thing the first time I watched this episode. It seemed like something significant was going on that I couldn't put my finger on. This time those Costa Mesa fires on TV really caught my attention- and it helped that I was just writing an essay on Inferno! But let me see what HASN'T been discussed yet...<br /><br />A TWOP review mentioned that Tony had 7 flights of stairs to go down because of the broken elevator. Yeah, 7 is a significant number for lots of reasons, especially religious, but here's one more for ya. On a hunch I consulted wikipedia, and guess what Dante divided into 7 levels? Purgatorio. Excluding ante-Purgatory and Paradise. (The stuff at the bottom of the stairs and... what Tony can't get to.) <br /><br />On to the allegedly "random" monk-slap scene. As soon as the monks appeared, it fit perfectly in place with Tony trying to get out of Purgatory. You can tell he got worried when that Christian commercial (death, disease, and sin) came on, and he's getting more and more desperate because Christian heaven is looking kinda iffy for him. By the time he meets the monks he's thinking "hey maybe these guys can help me?" which sounds like contemplating other religions (e.g. Buddhism) and wondering if some other path could take him to "salvation". Not that Tony is necessarily literally thinking about becoming a Buddhist, but it appears Finnerty tried that (and messed up). That slap in the face basically tells Tony there's no quick fix- as in, no, you can't suddenly embrace Buddhism and get out of here. <br /><br />Tony was initially not too concerned about getting to heaven. But at the "conference entrance", he realizes that's not going to be so easy for him. At first I saw the name vs. driver's license problem as Tony having led sort of a double life, what with the killing people and sleeping around that he kept secret from most people. He feels free to have an affair with quasi-Melfi because "he's Kevin Finnerty". He figures out that he CAN fool some people with KF's cards, like hotel receptionists, but it won't get him out of Purgatory. Those helicopters- the helicopters of Heaven?- are keeping track of him and everything he does.<br /><br />After reading all the theories on "inFinnerty", though, it seems like KF's identity is a reminder of the infinite different paths Tony could've taken in his life. Possibly along with the car joke involving Infiniti's that made no sense to me otherwise. Aaaand at that point my brain fizzles out.
True Label: positive
Predicted Label: negative
--------------------------------------------------
Review: I was very disappointed with this series. It had lots of cool graphics and that's about it. The level of detail it went into was minimal, and I always got the feeling the audience was being patronized -- there was a lot of what seemed to me as "This is extremely cool but we're not going to explain it in further detail because you won't get it anyway. Let's just show you some pretty pictures to entertain you." The host would drop interesting-sounding words such as "sparticles" and "super-symmetry" without any attempt at explaining what it was. We had to look it up on Wikipedia.<br /><br />Furthermore, I know quite a bit about superstrings (for a layman) and I found their explanations were convoluted and could have been so much better. They could have chosen MUCH better examples to explain concepts, but instead, the examples they used were confusing and further obscured the subject.<br /><br />Additionally, I got so sick of the repetitiveness. They could easily have condensed the series into one episode if they had cut out all the repetition. They must have shown the clips of the Quantum Café about 8 times. The host kept saying the same things over and over and over again. I can't remember how many times he said "The universe is made out of tiny little vibrating strings." It's like they were trying to brainwash us into just accepting "superstrings are the best thing since sliced bread."<br /><br />Finally, the show ended off with an unpleasant sense of a "competition" between Fermilab and CERN, clearly biased towards Fermilab. This is supposed to be an educational program about quantum physics, not about whether the US is better than Europe or vice versa! I also felt that was part of the patronizing -- "Audiences need to see some conflict to remain interested." Please. Give me a little more credit than that.<br /><br />Overall, 2 thumbs down :-(
True Label: negative
Predicted Label: negative
--------------------------------------------------
Review: The first 30 minutes of Tinseltown had my finger teetering on the remote, poised to flick around to watch something else. The premise of two writers, down on their luck, living in a self-storage-space "bin" was mildly amusing, but, painfully bland.<br /><br />The introduction of the character, played by Joe Pantoliano - the big deal movie guy, that lives in the park and sleeps in a lavatory, offered hope and I decided to give it a few more minutes. And then a few more until Kristy Swansons introduction as a budding film director & borderline nymphomaniac, added a bit of spice. Her solid acting performance raised her presence above and beyond just a very welcome eye-candy inclusion.<br /><br />Ultimately, the obvious low-budget impacts on the film with poorly shot scenes, stuttured pace and slapstick handling of certain moments. Some of my favourite movies of all time have been low budget, Whithnail & I being one that also deals with 2 guys with a dream, but down on their luck.<br /><br />However, for my money, the actors save Tinseltown from the "Terrible movie" archives and just about nudges it into the "could have been a cult movie" archives. I laughed out loud at some of the scenes involving Joe Pantoliano's character. In particular, the penultimate scenes in the terribly clichéd, but still funny, rich-but-screwed-up characters house, where the story unravels towards it's final moments.<br /><br />I can see how Tinseltown was a great stage play and while the film-makers did their best to translate this to celluloid, it simply didn't work and while I laughed out loud at some of scenes and one liners, I think the first 30 minutes dulled my senses and expectations to such a degree I would have laughed at anything.<br /><br />Unless you're stuck for a novelty coffee coaster, don't pick this up if you see it in a bargain bucket.
True Label: negative
Predicted Label: negative
--------------------------------------------------"""
