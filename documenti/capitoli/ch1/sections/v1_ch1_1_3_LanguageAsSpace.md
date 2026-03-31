# Chapter 1 — Meaning without measure

## 1.3 Language as space: from “meaning is use” to the distributional hypothesis


The figure who makes this convergence explicit is John Rupert Firth. In his “Synopsis of Linguistic Theory, 1930-1955,” Firth proposed a programme for the study of meaning that begins with a declaration of method: “By regarding words as acts, events, habits, we limit our inquiry to what is objective and observable in the group life of our fellows.” The resonance with Wittgenstein is not accidental. Firth had read the Philosophische Untersuchungen and cited them directly. His central claim was that meaning is not a referential property attached to a word but “a complex of contextual relations”: the meaning of a linguistic form is constituted by the contexts in which it appears, at multiple levels of analysis, from phonology to syntax to collocation.[^37][^38][^39]

It is at the collocational level that Firth produced the formulation that would become the most quoted sentence in distributional semantics. Discussing the habitual combinations in which words appear, Firth observed that the meaning of a word such as ass is inseparable from the company it keeps: “you silly—,” “he is a silly—,” “don’t be such an—.” From this he drew the general principle: “You shall know a word by the company it keeps!” The formulation is colloquial, but the epistemological content is precise. If meaning is constituted by habitual patterns of co-occurrence, and if those patterns are empirically observable, then meaning is not hidden in the private mental states of speakers. It is, in principle, accessible to systematic observation.[^40]

Firth’s contemporary, the American structural linguist Zellig Harris, arrived at a compatible conclusion through a more formal route. In his 1954 article “Distributional Structure,” Harris set out to determine whether language possesses a structure that can be described entirely in terms of the relative occurrence of its elements, “without intrusion of other features such as history or meaning.” His answer was affirmative. The distribution of an element, defined as “the sum of all its environments,” constitutes a structure that “really holds in the data investigated.” The question Harris then posed was whether this distributional structure bears any relation to meaning. His answer, cautious but consequential, was that “difference of meaning correlates with difference of distribution.” Two words that differ in meaning will tend to differ in the environments in which they appear; two words that share distributional environments will tend to share aspects of meaning.[^41][^42][^43]

The claim is not that meaning is distribution. Harris was careful to avoid the reductive identification. But the correlation he identified implies a procedure: if differences of meaning are systematically reflected in differences of distribution, then one can use the latter to discover the former. Distribution becomes the observable correlate of the unobservable phenomenon of meaning.

The theoretical ancestry of this insight is deeper than Harris himself made explicit. Magnus Sahlgren, in a careful reconstruction of the intellectual genealogy, has traced the distributional hypothesis back through Bloomfield and Harris to Ferdinand de Saussure. The Cours de linguistique générale (1916) had introduced the concept of valeur: the value of a linguistic sign is not its referential content but its position within the system of signs. A sign has a valeur “only by virtue of being different from the other signs.” In the language itself, Saussure argued, “there are only differences.” The example is well known: the French mouton and the English sheep may refer to the same animal, but they do not have the same valeur, because English draws a distinction between sheep and mutton that French does not. Meaning, for Saussure, is differential: it arises from the relations a sign entertains with all the other signs in the system, not from a direct link between word and world.[^44][^45]

This differential conception of meaning is the deep foundation of the distributional hypothesis. If meaning is constituted by relations of difference within a system (Saussure), and if those relations manifest themselves as patterns of distributional occurrence (Harris), then the observation of distributional patterns is an indirect observation of semantic structure. The distributional hypothesis, in its refined form, states that “a distributional model accumulated from co-occurrence information contains syntagmatic relations between words, while a distributional model accumulated from information about shared neighbors contains paradigmatic relations between words.” Co-occurrence captures which words combine (the syntagmatic axis); shared context captures which words are interchangeable (the paradigmatic axis). Both are dimensions of meaning in the Saussurean sense.[^46]

The philosophical chain, then, is this: meaning is use (Wittgenstein); use manifests as observable patterns of co-occurrence (Firth); those patterns exhibit a distributional structure that correlates with semantic structure (Harris); and the theoretical foundation for this correlation lies in the differential, relational nature of linguistic meaning itself (Saussure). None of these thinkers had the computational tools to exploit the convergence. But the conceptual apparatus was in place.


### The prehistory of meaning as space


Before the computational revolution made it possible to construct high-dimensional semantic spaces from distributional data, one attempt had already been made to subject meaning to spatial representation. In 1957, the same year in which Firth published his “Synopsis,” Charles Osgood, George Suci, and Percy Tannenbaum published The Measurement of Meaning. The book opened with a diagnostic that could serve as the epigraph of the present chapter: “Apart from the studies to be reported here, there have been few, if any, systematic attempts to subject meaning to quantitative measurement.”[^47]

Osgood’s method, the semantic differential, asked subjects to rate concepts on a series of bipolar scales (good-bad, strong-weak, active-passive, and dozens of others). Factor analysis applied to the resulting data consistently yielded three principal dimensions, which Osgood labelled evaluation, potency, and activity. A concept could thus be located in a three-dimensional semantic space, and the distance between two concepts in that space could be computed and compared. The geometric intuition was correct: meaning can be represented as position in a space, and semantic similarity can be measured as spatial proximity. What was missing was both the dimensionality and the data. Three dimensions, however robust across cultures, cannot capture the complexity of an entire vocabulary. And the data, gathered through controlled experiments with human subjects, could not scale beyond a few hundred concepts.

The gap between Osgood’s intuition and its full realisation would persist for more than half a century. What closed it was not a conceptual advance but an engineering one.


### The computational revolution


In 2013, Tomas Mikolov and his collaborators at Google published a paper that transformed the distributional hypothesis from a theoretical principle into a practical instrument. Their model, word2vec, learned vector representations of words from large text corpora by training a shallow neural network to predict either a word from its context (the Continuous Bag-of-Words architecture) or the context from a word (the Skip-gram architecture). The resulting vectors exhibited a remarkable property: semantic relations between words were preserved as geometric relations between vectors. The now-iconic demonstration was that vector(“King”) − vector(“Man”) + vector(“Woman”) yielded a vector closest to vector(“Queen”). The relation between “King” and “Man” (gender) was captured as a direction in the vector space, and this direction could be composed algebraically with other relations. Meaning, at least in its relational aspect, had become computable.[^48][^49]

The following years brought a series of architectural advances. Pennington, Socher, and Manning (2014) developed GloVe, which combined local context windows with global co-occurrence statistics. Vaswani and colleagues (2017) introduced the Transformer architecture, whose self-attention mechanism allowed the model to attend selectively to different parts of the input, enabling richer contextual representations. Devlin and colleagues (2019) built upon the Transformer to produce BERT, a model that generated contextual representations: unlike word2vec, where each word has a single fixed vector, BERT assigns different vectors to the same word in different contexts, capturing polysemy and contextual modulation. Reimers and Gurevych (2019) further adapted BERT to produce Sentence-BERT, which generates dense vector representations not of individual words but of entire sentences, making it possible to represent complex semantic units as points in a shared embedding space.[^50][^51][^52][^53]

The technical details of these architectures belong to Chapter 2. What matters here is the epistemological import. Each of these models is an implementation, at increasing levels of sophistication, of the distributional hypothesis: meaning is captured by observing patterns of co-occurrence in large corpora. The resulting representations are geometric: every word, every sentence, occupies a position in a high-dimensional space, and the distances and directions in that space encode semantic relations. Osgood’s three-dimensional semantic space has become a space of 768 or 1024 dimensions; his hundreds of experimentally rated concepts have become hundreds of thousands of words represented automatically from billions of sentences. The intuition is the same; the scale is transformed.


### Embedding spaces as objects of study


A decisive shift occurred when researchers began to treat embedding spaces not as tools for performing tasks, but as objects of study in their own right. The standard use of word embeddings in natural language processing is instrumental: the model is trained to perform a downstream task (translation, classification, retrieval), and its internal representations are means to that end. The epistemic turn consists in inverting the question: instead of asking “what can the model do?”, one asks “what does the model know, and where does this knowledge come from?”

The answer, demonstrated by a series of studies in the social sciences, is that embedding spaces encode the cultural structure of the communities that produced the training text. Kozlowski, Taddy, and Evans (2019) showed that dimensions induced by word differences in embedding spaces, such as man − woman, rich − poor, black − white, “closely correspond to dimensions of cultural meaning.” Projecting occupation names onto a gender dimension, they found that traditionally feminine occupations (“nurse,” “nanny”) were positioned at one end and traditionally masculine occupations (“engineer,” “lawyer”) at the other. The embedding space did not merely record which words co-occur; it had absorbed the social structure of gender as a geometric property of its internal representation. Kozlowski and colleagues concluded with a call for “high-dimensional theorizing” of meanings, identities, and cultural processes, arguing that the complexity of cultural phenomena requires instruments with corresponding dimensionality.[^54][^55]

Caliskan, Bryson, and Narayanan (2017), in a paper published in Science, provided perhaps the most striking demonstration. They developed the Word Embedding Association Test (WEAT), a statistical procedure analogous to the Implicit Association Test used in social psychology, and applied it to GloVe embeddings trained on a standard web corpus. The result was that every human bias tested, from the innocuous (flowers are more pleasant than insects) to the socially consequential (European-American names are more associated with “pleasant” attributes than African-American names), was replicated in the embedding space. Their conclusion was direct: “text corpora contain recoverable and accurate imprints of our historic biases.” The word embeddings, they noted, “know” these associations “with no direct experience of the world, and no representation of semantics other than the implicit metrics of words’ co-occurrence statistics with other nearby words.” The model is not a mind; it is a statistical summary of a community’s linguistic practices. And that summary, precisely because it is statistical rather than interpretive, is systematic, replicable, and measurable.[^56][^57]

Garg, Schiebinger, Jurafsky, and Zou (2018) extended the analysis diachronically, training embeddings on text corpora spanning more than a century and tracking the evolution of gender and ethnic stereotypes over time. They validated their measurements against historical US Census data, showing that changes in embedding bias tracked changes in occupational demographics. The embedding was not merely a record of language; it was a “quantitative lens” through which real-world social phenomena could be observed indirectly through their linguistic traces.[^58][^59]

Two further programmes deserve mention. Hamilton, Leskovec, and Jurafsky (2016) used diachronic word embeddings to track semantic change over centuries, demonstrating that the laws of semantic drift are statistically regular and that embedding models can detect shifts invisible to the naked philological eye. Bolukbasi, Chang, Zou, Saliga, and Kalai (2016) approached the same phenomenon from the opposite direction, developing methods to “debias” word embeddings by removing gender-associated components from the vector space. The debiasing programme treats the cultural information encoded in embeddings as a defect to be corrected. From the perspective of the present inquiry, the same information is not a defect but a datum: it is precisely because embeddings absorb the semantic structure of their training community that they can serve as instruments of measurement for that community’s conceptual organisation.[^60][^61]

These social-science applications establish a precedent that the present thesis extends to the legal domain. If embedding spaces can serve as instruments of measurement for cultural phenomena such as gender stereotypes (Caliskan), class associations (Kozlowski), and historical semantic change (Hamilton, Garg), the question for legal methodology is whether they can serve the same function for legal meaning. Is the semantic organisation of legal concepts, as it emerges from the practices of a legal community, observable in the geometry of an embedding space trained on that community’s texts?


### The comparability problem


The question, however, immediately encounters an obstacle. If different embedding models produce different spaces, with different dimensionalities and arbitrarily oriented axes, the vectors generated by one model cannot be directly compared to the vectors generated by another. The representations are incommensurable as mathematical objects. How, then, can one compare the semantic organisation captured by independently trained models? Can the structure of one space be compared to the structure of another, even when the spaces themselves share no common coordinates? This is the problem that the next section addresses.



---

[^37]: FIRTH, J. R., “A Synopsis of Linguistic Theory, 1930-1955,” in Studies in Linguistic Analysis, Basil Blackwell, Oxford 1957, p. 2.


[^38]: Firth cites Philosophische Untersuchungen explicitly on p. 11 (fn 4: “See L. Wittgenstein, Philosophical Investigations, Oxford, Blackwell, 1953, pp. 53, 61, 80, 81”). The citation confirms that the Wittgenstein-Firth convergence is not a retrospective scholarly construction but a connection attested in Firth’s own text.


[^39]: Ibid., p. 6: “Each function will be defined as the use of some language form or element in relation to some context. Meaning, that is to say, is to be regarded as a complex of contextual relations.”


[^40]: Ibid., p. 11. The full passage reads: “As Wittgenstein says, ‘the meaning of words lies in their use.’ The day to day practice of playing language games recognizes customs and rules. […] You shall know a word by the company it keeps!” Firth’s preceding discussion (p. 12) introduces the concept of “mutual expectancy” between co-occurring words, anticipating the statistical notion of co-occurrence probability that underlies all distributional models.


[^41]: HARRIS, Z. S., “Distributional Structure,” Word, 10(2-3), 1954, p. 146.


[^42]: Ibid., pp. 146, 149. The distribution of an element is “the sum of all its environments,” where “an environment of an element A is an existing array of its co-occurrents.”


[^43]: HARRIS, Z. S., *Mathematical Structures of Language*, Wiley, New York 1968, p. 12; and “Distributional Structure” (1954, reprinted 1970), p. 786: “if we consider words or morphemes A and B to be more different in meaning in A and C, then we will often find that the distributions of A and B are more different than the distributions of A and C.” The formulation is quoted and discussed in SAHLGREN, M., “The distributional hypothesis,” Rivista di Linguistica, 20(1), 2008, p. 36.


[^44]: SAHLGREN, M., “The distributional hypothesis,” Rivista di Linguistica, 20(1), 2008, pp. 33-53, spec. 4 (“The origin of differences”), pp. 38-40.


[^45]: SAUSSURE, F. de, *Cours de linguistique générale*, ed. Ch. Bally and A. Sechehaye, Payot, Lausanne-Paris 1916. «Dans la langue il n’y a que des différences.» The passage is at p. 166 of the 1916 edition (p. 166-118 in the De Mauro critical edition, Payot 1967). English transl.: *Course in General Linguistics*, transl. W. Baskin, Philosophical Library, New York 1959. The concept of valeur is developed in Part II, Ch. IV. On the mouton/sheep example, see Sahlgren (2008), p. 38.


[^46]: SAHLGREN (2008), p. 40. This is Sahlgren’s “refined distributional hypothesis,” which specifies what kind of semantic information different types of distributional models capture: co-occurrence models (e.g. LSA, word2vec Skip-gram) encode syntagmatic relations (which words combine); shared-neighbor models encode paradigmatic relations (which words are interchangeable).


[^47]: OSGOOD, C. E., SUCI, G. J., and TANNENBAUM, P. H., *The Measurement of Meaning*, University of Illinois Press, Urbana 1957, p. 1. The three dimensions consistently identified by factor analysis, Evaluation, Potency, and Activity (EPA), were found to be cross-culturally robust, suggesting a universal substrate of affective meaning. See Ch. 2, “The Dimensionality of the Semantic Space.”


[^48]: MIKOLOV, T., CHEN, K., CORRADO, G., and DEAN, J., “Efficient Estimation of Word Representations in Vector Space,” in Proceedings of the ICLR 2013 Workshop, 2013, pp. 4-5. The CBOW architecture predicts the target word from its surrounding context; the Skip-gram architecture predicts the surrounding context from the target word. Both implement the distributional hypothesis at scale.


[^49]: Ibid., p. 2. The result was originally reported in T. MIKOLOV, W.-T. YIH, and G. ZWEIG, “Linguistic Regularities in Continuous Space Word Representations,” in Proceedings of NAACL-HLT 2013, pp. 746-751.


[^50]: PENNINGTON, J., SOCHER, R., and MANNING, C. D., “GloVe: Global Vectors for Word Representation,” in Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2014, pp. 1532-1543.


[^51]: VASWANI, A. et al., “Attention Is All You Need,” in Advances in Neural Information Processing Systems 30 (NeurIPS 2017), 2017.


[^52]: DEVLIN, J., CHANG, M.-W., LEE, K., and TOUTANOVA, K., “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” in Proceedings of NAACL-HLT 2019, pp. 4171-4186.


[^53]: REIMERS, N., and GUREVYCH, I., “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks,” in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2019, pp. 3982-3992.


[^54]: KOZLOWSKI, A. C., TADDY, M., and EVANS, J. A., “The Geometry of Culture: Analyzing Meaning through Word Embeddings,” American Sociological Review, 84(5), 2019, pp. 905-949, at p. 905.


[^55]: Ibid., p. 905: “the success of these high-dimensional models motivates a move towards ‘high-dimensional theorizing’ of meanings, identities and cultural processes.”


[^56]: CALISKAN, A., BRYSON, J. J., and NARAYANAN, A., “Semantics derived automatically from language corpora contain human-like biases,” Science, 356(6334), 2017, pp. 183-186, at p. 183.


[^57]: Ibid., p. 184. The passage continues: “Notice that the word embeddings ‘know’ these properties of flowers, insects, musical instruments, and weapons with no direct experience of the world.” The absence of referential grounding makes the result more, not less, significant for the thesis’s argument: all semantic structure in the model derives from distributional patterns alone.


[^58]: GARG, N., SCHIEBINGER, L., JURAFSKY, D., and ZOU, J., “Word Embeddings Quantify 100 Years of Gender and Ethnic Stereotypes,” Proceedings of the National Academy of Sciences, 115(16), 2018, pp. E3635-E3644, at pp. E3637-E3638.


[^59]: Ibid., p. E3636: “Embeddings thus provide an important new quantitative metric which complements existing (more qualitative) linguistic and sociological analyses of biases.” The formulation captures the epistemological position of the present thesis exactly: embedding spaces do not replace hermeneutic interpretation but supplement it with measurable structure.


[^60]: HAMILTON, W. L., LESKOVEC, J., and JURAFSKY, D., “Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change,” in Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016), 2016, pp. 1489-1501.


[^61]: BOLUKBASI, T., CHANG, K.-W., ZOU, J. Y., SALIGRAMA, V., and KALAI, A. T., “Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings,” in Advances in Neural Information Processing Systems 29 (NeurIPS 2016), 2016, pp. 4349-4357.

