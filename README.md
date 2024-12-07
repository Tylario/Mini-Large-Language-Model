# Text Prediction and Generation with r/Jokes Data
**Author:** Tyler Hudson

## 1. Introduction
The goal of this project is to generate text by predicting the next character in a sequence. I originally wanted to use my personal social media messages as training data. However, due to access restrictions, I chose to work with data scraped from the r/jokes subreddit. This process will involve predicting values based on the previous sequence of values. We can continuously generate text by chaining predicted characters together using a sliding window technique. I am trying to create a miniature version of a large language model (LLM), capable of text generation based on patterns learned from training data.

This project uses two different methods for prediction: a neural network and a Markov Chain model. Neural networks are foundational to most modern LLMs, making this an ideal starting point. The Markov Chain, by contrast, is a simpler statistical approach, often used for predicting the next character based on probabilities derived from prior sequences.

## 2. Data
The dataset for this project comes from a GitHub repository by Taivo Pungas (@taivop) and Josh H (@stampyzfanz), featuring jokes scraped from Reddit’s r/jokes. It is publicly available [here](https://github.com/taivop/joke-dataset/blob/master/reddit_jokes.json).

## 3. Preprocessing
[Data Preprocessing Script](https://github.com/Tylario/Mini-Large-Language-Model/blob/main/dataProcessor.py)

The preprocessing stage included cleaning text by removing special characters and normalizing whitespace, combining the title and body of each joke, and outputting a single text file with all jokes concatenated.

- **Total number of characters:** 1208098
- **Unique characters:** 74 (` !"\'(),-.0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz`)
- **Character frequencies:**
  - ' ': 234694 (19.43%)
  - 'e': 108532 (8.98%)
  - 't': 77032 (6.38%)
  - 'a': 74520 (6.17%)
  - 'o': 68701 (5.69%)
  - 'n': 59589 (4.93%)
  - 's': 58054 (4.81%)
  - 'i': 55549 (4.60%)
  - 'h': 53497 (4.43%)
  - 'r': 48237 (3.99%)
  - 'd': 38194 (3.16%)
  - 'l': 35872 (2.97%)
  - 'u': 26755 (2.21%)
  - 'm': 21760 (1.80%)
  - ...
  - 'Z': 66 (0.01%) 
  - 'Q': 57 (0.00%) 
  - 'X': 57 (0.00%)

**Sample Text from Dataset:**
> I walked into a PETA adoption center and the receptionist asked me what kind of dog I wanted Apparently "Whatever's low in cholesterol" was not the right answer. How did TV studios make words appear on screen before computers? Character actors! How do you find Will Smith in the snow? You follow his fresh prints. Remember when you were a kid and when you cried your parents said, "I'll give you a reason to cry"? I always thought they were gunna hit me, not that they were going to destroy the housing market 20 years later. Remember, you can't spell Valentine's Day without... ... Anal Destiny. Happy try not to jump off a bridge day! Oh and it's also Valentines day My boss said to me, "you're the worst train driver ever. How many have you derailed this year?" I said, "I'm not sure; it's hard to keep track." If I get a bird I'm naming it Trump cuz all they do is Tweet I was going to get a headjob for Valentines Day But my back was sore and I couldn't reach. A mother went into a coma after giving birth to twins When she woke up after 6 months and 3 days, the doctor told the mother: "While you were in a coma, we had your brother name your children. One is a boy, one is a girl." The mother, with a disappointed and angry look on her face told the doctor: "Why my brother? That guy is an idiot…

## 4. Methods
[Neural Network Implementation](https://github.com/Tylario/Mini-Large-Language-Model/blob/main/neuralNetwork.py)
[Markov Chain Implementation](https://github.com/Tylario/Mini-Large-Language-Model/blob/main/markovChain.py)

### 4.1 Neural Network Method
This network consists of multiple layers organized in a feed-forward configuration. The input layer contains 199 neurons that process one-hot encoded character vectors through flattened character sequences. The hidden layer structure comprises four layers with dimensions [199, 199, 199, 100] neurons, each using Rectified Linear Unit (ReLU) activation functions with layer normalization applied post-activation. The output layer contains 74 neurons, corresponding to the vocabulary size, and employs a softmax activation function to produce a probability distribution over the character space.

The forward propagation process follows the mathematical framework defined by Z[l] = W[l]A[l-1] + b[l] and A[l] = σ(Z[l]), where W[l] represents the weight matrix at layer l, A[l] denotes the activation output, b[l] is the bias vector, and σ(·) represents the activation function. Each layer's output serves as input to the following layer.

The backward propagation process computes gradients using the chain rule, where δ[l] = δ[l+1]W[l+1]ᵀ ∘ σ'(Z[l]) and the weight updates are calculated as ∂W[l] = A[l-1]ᵀδ[l]. Here, δ[l] represents the error term at layer l, ∘ denotes element-wise multiplication, and σ'(·) is the derivative of the activation function. Each layer's gradients serve as input to the previous layer's computation.

The weights in each layer are initialized using the Xavier scheme, which selects values from a uniform distribution scaled by the layer size. The model employs momentum-based gradient descent with an adaptive learning rate that decreases over time. The implementation includes several regularization mechanisms: L2 weight decay to penalize large weights, gradient norm clipping to prevent exploding gradients, and mini-batch processing to optimize convergence.

For text generation, the model uses a sequence from the training data as inputs. By chaining together these predictions, the network can continuously generate text.

### 4.2 Markov Chain Method
This model constructs a frequency map to record how often each sequence of words appears in the text and what usually follows each sequence. The text is split into words using a simple space-based tokenization (anything between spaces is considered a word, including punctuation). This information is then used to establish transition probabilities, showing the likelihood of encountering a particular subsequent word after a given sequence of words. For example, if the sequence "what do you" appears frequently in the training text and is often followed by "mean", the model will assign a high probability to "mean" being the next word when it encounters "what do you" during generation.

The model analyzes sequences ranging from one to eight words in length, calculating the frequency of following words for each sequence. These frequencies are then integrated into a weighted probability distribution, where shorter sequences are given greater weight. Temperature is applied to this weighted distribution to control the randomness of the text generation. A lower temperature skews the predictions toward more probable options, resulting in more deterministic output, while a higher temperature increases randomness and diversity in the generated text.

For text generation, the model starts with a random sequence from the training data and uses the transition probabilities to predict the next word. The prediction is made through a weighted random selection from the possible options. By chaining together these predictions, the network can continuously generate text.

## 5. Results
### 5.1 Neural Network Results
These are the hyperparameters:
```python
predictor = CharacterPredictor(
    input_size=199,
    hidden_layers=[199, 199, 199, 100],
    output_size=74,
    learning_rate=0.001,
    epochs=10,
    batch_size=128,
    decay_rate=0.1,
    weight_decay=0.0001
)
```

**Loss during training:**
| Batch | Loss   |
|-------|--------|
| 100   | 0.0135 |
| 200   | 0.0134 |
| 300   | 0.0133 |
| ...   | ...    |
| 4400  | 0.0126 |

**These are the results when choosing the next character based on the highest probability:**
(Include any relevant output or examples here.)

**These are the results when choosing the next character randomly based on weighted probability:**
> t tt ae a aea t t aa ae ee et o eat t ea e o a eot t e a t ta o e a oe ee e e ttt eeee e t t t ao ae e e aa tea atat e e eae t o e e o e ee eo eee e tea oo t e e o t o ee o te t t t otee e e t a et a et at te ee e aaato e ae o oe ee e oe t e t o t te

### 5.2 Markov Chain Results
**These weights determine the influence of sequences based on

 their length:**
```python
weights = {
    1: 0.1,         # Single word context
    2: 0.2,         # Two word context
    3: 0.1,         # Three word context
    4: 0.05,       # Four word context
    5: 0.025,     # Five word context
    6: 0.0125,   # Six word context
    7: 0.00625, # Seven word context
    8: 0.003125# Eight word context
}
```

**These are the results with temperature = 0.1:**
> . . I'll Never take her back .. !! Its a long time in the middle of the United States border... Ahmad, a Muslim American man, was a little boy in church. He had to go to the other day and found a place where the recycling rate is 98 rJokes I wanted to make it stop, say 'Cid Ood-Oov return,' and Voo-Doo Dick returns to the other and says, "DUDE! Did you call a gay version of Donald Trump? Trump only advocates the seizing of a sudden a man was sitting on the bus, the man is forced to the other day I asked my wife told me I was a baker. Then I would be rolling in pussy! Girlfriend is gone now, but she left her husband and the other day the man replies "I have a request," says the pilot, "could you please help me." The man says, "I don't want to have a good day sir" Said the man with a fork sticking out of his wife is in the next day the same question. The man replies, "I know, I don't have a little boy raises his hand. However, he was a little girl What do you call a gay men sound the

**These are the results with temperature = 1:**
> Joe bought a gay in your head) I'm so horny. I need help. Why did the hipster burn his tongue? He drank coffee before it was cool QUEEN Elizabeth and Dolly Parton die on the Russian doctor asks the blond and instantly filled with the bear: "I just holler out the door, Leroy! its time for his indoctrination to sex." The madam says, "Since this is all dear; let's go to like going for president But we will never have figured he was a good counsellor he immediately communicates the phone again, and as soon as an old one, now where is your mother? SON: I don't know, it was impossible to catch! He asks. "I showed him." I love her leg. The doctor to get a diagnosis. "Well, there's good news and some bad news?" And the parrot again explaining the situation. The pink gorilla that swung around for a while. The farmer says "I'll have sex?" Why is your last three men, an Englishman, an Irishman and says "I've been seeing this girl for 2 Jewish men - Abraham and Jacob - are stuck in the shorthand. She responded that it was "FUCKING RAAAAAAWWWWW!" A lot of stuff until a problem

**These are the results with temperature = 5:**
> . .Fuck Hans. A flood warning is always on, even after never being able to withdraw." "Do you know Who do you call A anorexic girl with a parrot on me? Boy: Of course. Lots! Girl: Have you ever told. I tried to treat me different because of my long and Bulbous nose." The old man, "Even easier" the back. This girl said she says we'll have none of them in Michelle. What did Nancy Sinatra say about her actor friend Christopher's custom-made footwear? These boots are made for walking by this, but you take. Why I got lost spice girl... Stupid Spice People told me is technically a plant? Because all of his profession decided to transfer out Bernie was very tired. At one of her students... A teacher is giving her kindergarten class a spelling had some cents knocked into me. I like my jokes how I got nothing to play with all decided which I did by wiping it on the very bottom up. After death, what is a lot like a Dragon Ball Z villain He has yet to kneel and instead turned his face in Mexico can foot the bill. The German doctor says " here's the other: "Man, this

## 6. Analysis
### 6.1 Neural Network Performance
The neural network implementation faced several significant challenges in terms of computational limitations, output quality, and model design. The error plateaued around 0.0126. The training process took approximately one hour even with a reduced dataset of 10,000 jokes, making it impractical to utilize the full dataset due to computational constraints. This limited training data ultimately resulted in poor model performance.

When using maximum probability selection for character generation, the model consistently defaulted to selecting spaces, producing blank output. Random weighted selection produced gibberish that merely reflected basic character frequencies. The model failed to learn any significant patterns or sentence structure from the training data.

### 6.2 Markov Chain Performance
The Markov Chain implementation showed considerably better results. The generated text maintained basic sentence structure and grammar. Different temperature settings produced slightly different results: at 0.1, the text was more coherent but somewhat repetitive; at 1.0, it achieved a balance between coherence and creativity; and at 5.0, the output became more random and diverse, though less coherent.

## Conclusions
This project explored two different approaches to text generation using Reddit jokes as training data: a neural network and a Markov Chain model. The results demonstrated the inherent challenges and trade-offs in text generation tasks, particularly when working with limited computational resources.

The neural network approach, while theoretically more sophisticated, faced significant practical limitations. The computational constraints forced me to use a reduced dataset and limited training, which resulted in poor model performance. Overall this model tended to output high-probability characters (spaces) when using maximum probability selection, and meaningless character sequences when using weighted random selection.

The Markov Chain model, despite being a simpler statistical approach, proved more effective. It successfully captured basic sentence structure and grammar, producing more coherent and readable output. The multi-order approach, which was my own idea, was very beneficial to producing better results.

The key learning from this project is that sophisticated neural approaches aren't always the best solution, particularly when working with limited resources. The Markov Chain's success shows that simpler, statistical methods can sometimes outperform more complex neural networks when dealing with constraints like computational power.

Through this project, I have gained valuable insights into the practical applications of machine learning techniques and the importance of model selection based on the available resources. This experience has boosted my confidence in my ability to implement a full machine learning pipeline and has affirmed my capability to handle complex computational tasks independently.

## Acknowledgements
This project was completed using the r/Jokes dataset as the sole data source. All assistance, including coding support, debugging, and suggestions for improvements, was obtained exclusively from ChatGPT and the Cursor text editor. I would also like to thank Minwoo "Jake" Lee, my instructor for the course, whose lectures were highly informative and greatly enhanced my understanding of the subject.

## Source Code
[GitHub Repository](https://github.com/Tylario/Mini-Large-Language-Model)
