import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

df = pd.read_csv('IMDB Dataset.csv')

texts = df['review'].values
labels = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0).values

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

tokenizer = Tokenizer(num_words=10000) 
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_len = 100  
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=max_len),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_split=0.2)

test_loss, test_acc = model.evaluate(X_test_pad, y_test)
print(f"Test Accuracy: {test_acc}")


indices = np.random.choice(len(X_test), size=10, replace=False)
sample_reviews = X_test[indices]
sample_labels = y_test[indices]

sample_reviews_texts = [texts[i] for i in indices] 

# Tokenize and pad sample reviews
sample_seq = tokenizer.texts_to_sequences(sample_reviews_texts)
sample_pad = pad_sequences(sample_seq, maxlen=max_len)

predictions = model.predict(sample_pad)
for review, true_label, prediction in zip(sample_reviews_texts, sample_labels, predictions):
    sentiment = 'positive' if prediction >= 0.5 else 'negative'
    true_sentiment = 'positive' if true_label == 1 else 'negative'
    print(f"Review: {review}\nTrue Label: {true_sentiment}\nPredicted Label: {sentiment}\n")


"""
Epoch 1/5
500/500 ━━━━━━━━━━━━━━━━━━━━ 86s 171ms/step - accuracy: 0.7359 - loss: 0.5076 - val_accuracy: 0.8611 - val_loss: 0.3296
Epoch 2/5
500/500 ━━━━━━━━━━━━━━━━━━━━ 87s 174ms/step - accuracy: 0.8957 - loss: 0.2659 - val_accuracy: 0.8641 - val_loss: 0.3167
Epoch 3/5
500/500 ━━━━━━━━━━━━━━━━━━━━ 279s 559ms/step - accuracy: 0.9304 - loss: 0.1850 - val_accuracy: 0.8596 - val_loss: 0.3523
Epoch 4/5
500/500 ━━━━━━━━━━━━━━━━━━━━ 86s 171ms/step - accuracy: 0.9514 - loss: 0.1382 - val_accuracy: 0.8553 - val_loss: 0.4780
Epoch 5/5
500/500 ━━━━━━━━━━━━━━━━━━━━ 95s 189ms/step - accuracy: 0.9647 - loss: 0.0965 - val_accuracy: 0.8506 - val_loss: 0.4655
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 32ms/step - accuracy: 0.8456 - loss: 0.4825
Test Accuracy: 0.8479999899864197
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 102ms/step
Review: This is a nice little movie with a nice story, that plays the most important role in the entire movie.<br /><br />It's a quite intriguing dramatic story, with also romance present in it. The story is being told slowly but this works out all too well for its build up. The characters are nice and portrayed nicely by its actors. Normally I'm not a too big fan of the Asian acting style but the acting in this movie was simply good.<br /><br />Of course the movie is quite different in its approach and style from other genre movies, produced in the west. In a way this movie is more advanced already with its approach than the western movies made during the same era.<br /><br />I only wished the movie its visual style would had been a bit better. For a movie that is considered a kind of an art-house movie this movie is certainly lacking in some well looking sequences. This was obviously a quite cheap movie to make and it got made quite generically. Not that this is a bad thing, it just prevent this movie from truly distinct itself and raising itself above the genre.<br /><br />But oh well, this movie is all about its well constructed story and characters that are in it. In that regard this movie most certainly does not disappoint.<br /><br />8/10
True Label: negative
Predicted Label: positive

Review: I don't know what some of you are smoking, but i suspect it's potent.<br /><br />To call Swept Away awful would be an insult to the very concept of terribleness. The acting is hideous and i'm not picking on Madonna here, we all know she's useless, but someone should have warned everyone else that her ailment is contagious. My back literally hurts from cringing so much at poorly delivered lines. The editing is so sloppy, it beggars description. The photography and composition (which in this era, competence should be a GIVEN for any film with a budget) are astonishingly inept, even the lighting is horrid and unnatural looking. These are BASIC elements of filmmaking, if you can't get them right, you should seek another line of work. It's as contrived as a grade 3 production of Snow White, except nowhere near as well made or interesting.<br /><br />The original film by Lina Wertmueller is a wonderful satire and metaphor, superbly acted and written, featuring breathtaking visuals - you can practically taste the sea salt and feel the windswept sand in your hair. The sexual tension feels real and immediate...those of you who found Guy Ritchie's version deplorable, should see it, it really is one of the landmarks of world cinema.<br /><br />Those of you who thought the remake is some kind of masterpiece should have your heads examined.
True Label: negative
Predicted Label: negative

Review: There are some wonderful things about this movie. Marion Davies could act, given the right property; she is wonderful in comedic roles. William Haines could act, and you can see why he was one of the screen's most popular leading men. (Until a potential scandal forced him from the business).<br /><br />The story is a bit trite, but handled so beautifully that you don't notice. King Vidor's direction is one of the principle reasons for this. The producer? The boy genius, Irving Thalberg.<br /><br />It's about movie making, and you get to see the process as it was done in 1928, the cameras, sets, directors directing and actors emoting. You get to see (briefly) some of the major stars of the day; even Charlie Chaplin does a turn as himself, seeking an autograph. You also catch glimpses of Eleanor Boardman, Elinor Glyn, Claire Windsor, King Vidor, and many others who are otherwise just names and old photographs.<br /><br />Please, even if you're not a fan of the silents, take the time to catch this film when you can. It's really a terrific trip back in time.
True Label: positive
Predicted Label: positive

Review: I didn't know whether to laugh or cry at this misrepresentation of Canadian history, particularly the disservice done to the history of the Mounted Police in the Yukon.<br /><br />I'll leave it to Pierre Berton, noted historian, born and raised in Dawson City Yukon, and author of the definitive history of the Klondike gold rush, Klondike: The Last Great Gold Rush, 1896-1899 to express my exasperation with this silly movie: <br /><br />The American idea of an untamed frontier, subdued by individual heroes armed with six-guns, was continued in The Far Country, another story about a cowboy from the American west - Wyoming this time - driving his herd of beef cattle into gold country. The picture is a nightmare of geographical impossibilities, but the real incongruity is the major assumption on which the plot turns  that there was only one mounted policeman in all of the Canadian Yukon at the time of the gold rush and that he could not deal with the lawlessness. When James Stewart and Walter Brennan reach the Yukon border with their cattle, the customs shack is empty.<br /><br />"Where is the constable? asks Brennan.<br /><br />"Up on the Pelly River. Trouble with the Chilkats," someone replies. He's got a real tough job, that constable. He patrols some ten or twenty thousand square miles. Sometimes he don't get home for two or three months at a time." <br /><br />The historical truth is that the Yukon Territory during the gold rush was the closest thing to a police state British North America has ever seen. The Northwest Mounted Police was stationed in the territory in considerable numbers long before the Klondike strike. They controlled every route into the Yukon and they brooked no nonsense. They collected customs duties, often over the wails of the new arrivals, made arbitrary laws on the spot about river navigation, and turned men back if they didn't have enough supplies, or if they simply looked bad. In true Canadian fashion, they laid down moral laws for the community. In Dawson the Lord's Day Act was strictly observed; it was a crime punishable by a fine to cut your wood on Sunday; and plump young women were arrested for what the stern-faced police called "giving a risqué performance in the theatre," generally nothing more than dancing suggestively on the stage in overly revealing tights.<br /><br />In such a community, a gunbelt was unthinkable. One notorious bad man from Tombstone who tried to pack a weapon on his hip was personally disarmed by a young constable, who had just ejected him from a saloon for the heinous crime of talking too loudly. The bad man left like a lamb but protested when the policeman, upon discovering he was carrying a gun told him to hand it over. "No man has yet taken a gun away from me," said the American. "Well, I'm taking it", the constable said mildly and did so, without further resistance. So many revolvers were confiscated in Dawson that they were auctioned off by the police for as little as a dollar and purchased as souvenirs to keep on the mantelpiece.<br /><br />In 1898, the big year of the stampede, there wasn't a serious crime  let alone a murder  in Dawson. The contrast with Skagway on the American side, which was a lawless town run by Soapy Smith, the Denver confidence man, was remarkable. But in The Far Country Dawson is seen as a community without any law, which a Soapy Smith character from Skagway  he is called Gannon in the picture  can easily control. (In real life, one of Smith's men who tried to cross the border had all his equipment confiscated and was frogmarched right back again by a mounted police sergeant).<br /><br />{in the movie the lone Mountie says} "Yes I'm the law. I represent the law in the Yukon Territory. About fifty thousand square miles of it."<br /><br />"Then why aren't there more of you?"<br /><br />"Because yesterday this was a wilderness. We didn't expect you to pour in by the thousands. Now that you're here, we'll protect you."<br /><br />"When?"<br /><br />"There'll be a post established here in Dawson early in May."<br /><br />"What happens between now and May? You going to be here to keep order?"<br /><br />"Part of the time."<br /><br />"What about the rest of the time?"<br /><br />"Pick yourselves a good man. Swear him in. Have him act as marshal"<br /><br />The movie Mountie leaves and does not appear again in the picture. His astonishing suggestion  that an American town marshal, complete with tin star, be sworn in by a group of townspeople living under British jurisprudence  is accepted. Naturally they want to make Jimmy Stewart the marshal; he clearly fits the part. But Stewart is playing the role of the Loner who looks after Number One and so another man is elected to get shot. And he does. Others get shot. Even Walter Brennan gets shot. Stewart finally comes to the reluctant conclusion that he must end all the shooting with some shooting of his own. He pins on the tin star and he and the bully, Gannon, blast away at each other in the inevitable western climax.<br /><br />To anybody with a passing knowledge of the Canadian north, this bald re-telling of the story passes rational belief. <br /><br />excerpt from Hollywood's Canada, by Pierre Berton, 1975.
True Label: negative
Predicted Label: negative

Review: <br /><br />Fourteen of the funniest minutes on celluloid. This short parody is at least as much a part of the Star Wars saga as Phantom Menace, and far more entertaining, if you ask me. Hardware Wars was the first in a long line of SW spoofs which form their own subgenre these days. I hate to describe it too much-it's so short that the premise is just about the whole thing. Suffice it to say that many of the most popular and familiar aspects of Star Wars have fun poked at them. Household appliances such as toasters and vacuum cleaners portray spaceships and robots, the Princess Anne-Droid character wears actual bread rolls on her head instead of the famous coils of braided hair, and Fluke Starbucker is even more of a dork than his original, if that's possible. Ernie Fosselius is one crazy son-of-a-buck-he's also the source of Porklips Now, the Apocalypse Now spoof.
True Label: negative
Predicted Label: negative

Review: Deodato brings us some mildly shocking moments and a movie that doesn't take itself too seriously. Absolutely a classic in it's own particular kind of way. This movie provides a refreshingly different look at barbarians. If you get a chance to see this movie do so. You'll definitely have a smile on your face most of the time. Not because it's funny or anything mundane like that but because it's so bad it goes out the other way and becomes good, though maybe not clean, fun.
True Label: positive
Predicted Label: negative

Review: I like this movie cause it has a good approach of Buddhism, for example, the way Buddhist use to care all kind of living things, combining some fancy and real situations; in some parts the photography is very good and a lot of messages about freedom, as the hawk episode, staying always focused in every moment, even in tough situations.. It has also funny situations as Swank's birthday and, talking this two times academy awards, her acting show us how the people who use to live in this kind of culture is trying to have a resistance behavior when Miyagi is taking her to a Buddhist temple, and how she, slowly, is changing her mind. And, of course, Pat Morita has been always great
True Label: positive
Predicted Label: positive

Review: My wife rented this movie and then conveniently never got to see it. If I ever want to torture her I will make her watch this movie. I've watched many movies with my 4 year old and I can take almost anything. Barney is refreshing after a shot of Quigley. <br /><br />The plot, dialog, cinematography, & acting were one step above (or equal to) a cheap porn film. I feel cheated out of $3.69 that we paid to rent it and then 90 minutes of my life I will never get back. I will say my 4 year old liked it, luckily it was a rental we had to return right away.<br /><br />I just hope that the younger actor's careers are not ruined from being in this movie.
True Label: positive
Predicted Label: negative

Review: The recent release of "Mad Dog Morgan" on Troma DVD is disappointing.This appears to be a censored print for television viewing. Some of the more violent scenes have been edited and portions of the colorful language have been removed. Anyone who viewed the film uncut will be mad as hell at this toxic DVD version. "Mad Dog Morgan" deserves to be released on DVD in the original theatrical cut. However, even as released on DVD, the film is still one of the better depictions of bushranger life in nineteenth century Australia. After having toured the Old Melbourne Gaol, with death masks of convicts on display, it is "Mad Dog Morgan" that comes to mind.
True Label: negative
Predicted Label: negative

Review: Natalie Wood portrays Courtney Patterson, a polio disabled songwriter who attempts to avoid being victimized as a result of involvement in her first love affair, with her partner being attorney Marcus Simon, played tepidly by Wood's real-life husband, Robert Wagner. The film is cut heavily, but the majority of the remaining scenes shows a very weak hand from the director who permits Wagner to consistently somnambulate, laying waste to a solid and nuanced performance from Wood, who also proffers a fine soprano. The script is somewhat trite but the persistent nature of Wagner's dramatic shortcoming is unfortunately in place throughout, as he is given a free hand to impose his desultory stare at Wood, which must be discouraging to an actress. The progression of their relationship is erratically presented and this, coupled with choppy editing, leads the viewer to be less than assured as to what is transpiring, motivation being almost completely ignored in the writing. Although largely undistinguished, the cinematography shines during one brief scene when Wood is placed in a patio and, following the sound of a closing door, remains at the center while the camera's eye steadily pulls away demonstrating her helplessness and frailty. More controlled direction would have allowed the performers, even the limp Wagner, to scale their acting along the lines of an engaging relationship; as it was released, there is, for the most part, an immense lack of commitment.
True Label: positive
Predicted Label: negative

Process finished with exit code 0
"""
