# Chapter 1 — Meaning without measure

## 1.4 Space as legal instrument: second-order isomorphism and the measurability thesis


### The problem of cross-space comparison


The concept that Shepard and Chipman introduced requires a precise statement of the problem it solves. As the preceding section observed, different embedding models produce different spaces. One model may generate vectors in 768 dimensions; another, trained on a different corpus or with a different architecture, may produce vectors in 1024 dimensions. These spaces are not merely different in size. Their axes are artifacts of the training process, determined by random initialisation, corpus composition, and optimisation dynamics. No dimension in one space corresponds to any dimension in the other. A vector representing a given concept in the first model and a vector representing the same concept in the second are, as mathematical objects, incommensurable. Computing a distance between them would be like subtracting a temperature in Celsius from a weight in kilograms: the operation is arithmetically possible but semantically void.

This incommensurability has a methodological consequence. If one wished to compare how legal concepts are organised across different linguistic communities, or across different corpora within the same language, the natural approach would be to embed all terms in a single shared space (for instance, a multilingual model). But such models achieve cross-lingual alignment through their training procedure: parallel corpora, translation objectives, or shared subword vocabularies force different languages into geometric correspondence. The correspondence is imposed by the architecture, not discovered from the data. For any inquiry that seeks to determine whether two representational systems organise their concepts similarly or differently, this is methodologically self-defeating: a model that presupposes commensurability as a training condition cannot be used to test it.[^62]

The alternative is to keep the spaces separate, each reflecting only the distributional patterns of its own corpus and community, and to find a method of comparison that works across incommensurable spaces. But this immediately raises the problem it was designed to solve on principled grounds: if the spaces are separate, how can they be compared at all?

The answer lies in a distinction between two kinds of comparison. One can compare objects across spaces (impossible, as we have seen), or one can compare structures across spaces. What is needed is a method that extracts, from each space, a description of its internal organisation, and then compares those descriptions. Such a method must be indifferent to dimensionality, axis alignment, and the absolute position of vectors. It must attend only to the pattern of relations among the points within each space. This is precisely what the concept of second-order isomorphism provides.


### Shepard and Chipman: second-order isomorphism


In 1970, the psychophysicist Roger Shepard and his student Susan Chipman published a study on the internal representation of the shapes of the United States in human memory. The question they posed was deceptively simple: when a person imagines the shape of a state (Colorado, Texas, Florida), does the mental representation resemble the actual shape? The deeper question, however, was about the nature of representation itself.[^63]

The standard assumption, which Shepard and Chipman called first-order isomorphism, holds that a representation is adequate to the extent that it structurally resembles the object it represents. A map is good if it looks like the territory. A mental image of a square is faithful if it is, in some neural sense, square. But this assumption, as they recognised, is both empirically implausible and theoretically unnecessary. The mental representation of a square need not be square. What matters is something else entirely:


> The crucial step consists in accepting that the isomorphism should be sought, not in the first-order relation between (a) an individual object, and (b) its corresponding internal representation, but in the second-order relation between (a) the relations among alternative external objects, and (b) the relations among their corresponding internal representations.[^64]


The shift is from objects to relations. A representation is faithful not when it resembles its referent, but when the pattern of relations among representations mirrors the pattern of relations among referents. Shepard and Chipman offered an intuitive illustration: “although the internal representation for a square need not itself be square, it should (whatever it is) at least have a closer functional relation to the internal representation for a rectangle than to that, say, for a green flash or the taste of persimmon.” What is preserved is not the individual shape, but the relational structure: square is closer to rectangle than to persimmon, in both the external world and the internal representation.[^65]

The principle translates directly into the legal domain. The embedding of “due process” need not look like due process (whatever that would mean for a 768-dimensional vector). But it should be closer to the embedding of “fair trial” than to the embedding of “eminent domain.” If this relational structure is preserved, the representation, despite bearing no first-order resemblance to the legal concept, is informationally faithful in the sense that matters: it preserves the geometry of conceptual relations.

Shimon Edelman, extending Shepard’s framework into a general theory of neural representation, stated the principle with maximal economy: the task of a representational system is to “represent similarity between shapes, not the geometry of each shape in itself.” For the present argument, the implication is precise: what we seek to compare across legal traditions is not the geometry of individual concept-vectors (which is meaningless across spaces) but the similarity structure among all concept-vectors within each space. This similarity structure is, by definition, a second-order object. And it can be extracted, formalised, and compared.[^66]


### Kriegeskorte: Representational Similarity Analysis


The concept of second-order isomorphism remained, for nearly four decades, a theoretical principle without a standardised quantitative framework. The framework arrived in 2008, when Nikolaus Kriegeskorte, Marieke Mur, and Peter Bandettini proposed Representational Similarity Analysis (RSA) as a method for comparing neural representations across brains, species, and computational models. Kriegeskorte and colleagues explicitly acknowledged the intellectual debt: “One concept at the core of our approach is that of second-order isomorphism (Shepard and Chipman, 1970), i.e., the match of dissimilarity matrices.”[^67][^68]

The central object of RSA is the Representational Dissimilarity Matrix (RDM). For a set of N concepts represented in a given space, the RDM is an N x N symmetric matrix whose entry (i, j) records the dissimilarity between the representations of concepts i and j. The diagonal is zero by definition (each concept is identical to itself); the off-diagonal entries capture the full pattern of pairwise distances. As Nili and colleagues put it in their methodological codification: “Each off-diagonal value indicates the dissimilarity between the activity patterns associated with two different stimuli. The dissimilarities can be interpreted as distances between points in the multivariate response space. The RDM thus describes the geometry of the arrangement of patterns in this space.”[^69]

The decisive property of the RDM is what it discards. Kriegeskorte and colleagues emphasised that RDMs “abstract from the spatial layout of the representations. They are indexed (horizontally and vertically) by experimental condition and can thus be directly compared between brain and model. What we are comparing, intuitively, is the represented information, not the activity patterns themselves.” This abstraction is precisely what dissolves the cross-space comparison problem. One embedding space may have 768 dimensions; another may have 1024. Their RDMs, however, are both N x N matrices indexed by the same set of concepts. Dimensionality, axis alignment, and absolute vector position have been abstracted away. What remains is the relational structure, and this structure can be compared.[^70]

The comparison is operationalised as a correlation between the upper triangles of two RDMs. The choice of the Spearman rank correlation avoids assuming a linear relationship between the dissimilarity scales of the two spaces, a prudent choice when comparing representations from fundamentally different systems. Statistical significance is assessed by a permutation test: the concept labels of one RDM are randomly shuffled, the correlation is recomputed, and the procedure is repeated thousands of times to generate a null distribution. If the observed correlation exceeds the vast majority of permuted correlations, one concludes that the two spaces share relational structure beyond what chance would produce.[^71][^72]

Diedrichsen and Kriegeskorte subsequently demonstrated that this approach rests on a rigorous mathematical foundation. They showed that “the multivariate second moment of the activity profiles fully defines the representational geometry and with it all the information that can linearly or nonlinearly decoded.” The RDM, in other words, is not a convenient summary that discards information for the sake of comparability. It is a complete characterisation of the representational geometry: everything that can be known about the relational structure of a space is encoded in its pairwise distance matrix. For the present thesis, this result is essential. It means that the move from vectors to distances, which is necessary to solve the cross-space comparison problem, entails no loss of geometric information.[^73][^74]

The statistical apparatus of RSA, including the specific implementation choices adopted in this thesis, is developed in 2.4. What matters at this stage is the conceptual architecture: second-order isomorphism provides the principle; the RDM provides the formalisation; the Spearman correlation and permutation test provide the inference. Together, they constitute a framework for comparing the internal organisation of any two representational spaces, regardless of their dimensionality, training regime, or physical substrate. The framework was built for comparing brains and computational models. It applies, without modification, to comparing the embedding spaces of different legal traditions.


### The measurability thesis


The argument of this chapter can now be drawn together. Legal meaning, as the jurisprudential tradition from Savigny to Ehrlich has recognised, is a collective emergent phenomenon: it arises from the practices of a legal community, not from the intention of a single legislator (1.2). Meaning, as the philosophical and linguistic traditions from Wittgenstein to Harris have shown, is constituted by patterns of use, and these patterns manifest as distributional regularities that are, in principle, empirically observable (1.2, 1.3). Embedding models, trained on large corpora, capture these distributional regularities as geometric structure: legal concepts become points in high-dimensional spaces, and the distances among them encode semantic relations (1.3). And the relational structure of these spaces can be extracted, formalised, and compared across independently constructed models through the framework of second-order isomorphism and Representational Similarity Analysis (1.4).

From these premises, a thesis follows:


> The relational organisation of legal concepts in embedding spaces constitutes an observable, measurable, and empirically testable object of study for legal science.


This is the measurability thesis. It asserts that legal science now has access to an object it has never had before: the geometric pattern of distances among legal concepts, as encoded in the distributional practices of a linguistic community and captured by a computational model trained on those practices. This object is observable (it can be extracted from any embedding space), measurable (it can be quantified as a dissimilarity matrix), and empirically testable (hypotheses about its structure can be assessed through permutation tests and bootstrap procedures).

Three clarifications are immediately necessary.

First, the thesis does not claim that everything about legal meaning is captured by geometric structure. What is measured is the relational organisation of concepts: which are near, which are far, which domains are cohesive, which are fragmented. The hermeneutic dimensions of legal meaning, the argumentative force of a provision, the institutional context of its application, the normative weight of a precedent, remain outside the instrument’s reach. Measurable does not mean exhaustive.

Second, the thesis does not claim that individual embeddings are meaningful in isolation. A single 768-dimensional vector tells us nothing interpretable about the concept it represents. What carries information is the pattern of relations. As Shepard and Chipman observed in their concluding discussion, meaningful knowledge about internal representations concerns “the relations between that internal representation and other internal representations”; what is known, in the end, are “properties of sets of stimuli, not properties of individual stimuli.” The object of measurement is always the set, never the element.[^75]

Third, the thesis locates the embedding space as a cultural object. Aceves and Evans, articulating a parallel argument for the social sciences, have stated the point with precision: “each vector within a word embedding model represents a concept and […] the entire embedding model represents the conceptual space of the social system that generated the textual data.” The embedding space of a legal tradition is not a neutral mathematical construct. It is a computational crystallisation of a community’s linguistic practices, and the geometric structure it exhibits is, in this sense, the structure of that community’s conceptual organisation of law. Bhupatiraju, Chen, and Venkataramanan have applied the geometric metaphor directly to legal texts, observing that embedding representations are “information-dense in the sense of retaining information about semantic content and meaning, while also being computationally tractable.” Their approach, however, remains applied: embeddings as tools for prediction and classification. The present thesis takes the further step of providing the epistemological framework, grounded in second-order isomorphism, that justifies treating the geometry of legal embeddings not merely as a useful tool but as a legitimate object of scientific inquiry.[^76][^77][^78]


### What becomes possible: instrument validation logic


If the measurability thesis is correct, a programme of empirical investigation becomes possible. But the programme must be structured not as an exploration of legal-cultural differences (which would be comparative law) but as the validation of a measurement instrument (which belongs to methodology). The logic of instrument validation requires answering three questions, in sequence.

Does the signal exist? The first question is whether legal meaning possesses non-random geometric structure in embedding spaces. If the distances among legal concepts were indistinguishable from random noise, no further analysis would be warranted. This is tested by comparing the observed RDM to a null distribution generated by permuting concept labels. If the permutation test rejects the null hypothesis, the embedding space contains a detectable legal-semantic signal.

Is the instrument reliable? The second question is whether the structure detected by one model is also detected by independently trained models. If the geometric pattern were an artifact of a single model’s training idiosyncrasies, it would not constitute a reliable measurement. Reliability is tested by computing RDMs from multiple models trained on different corpora and with different architectures, then assessing their mutual correlation.

Does the instrument have discriminating power? The third question is whether the instrument can detect differences where theory and institutional knowledge lead us to expect them. A thermometer that always reads the same temperature is useless, however precise. A measurement instrument for legal meaning must be capable of distinguishing structures that are expected to differ. This requires exposing the instrument to known or expected variation: for instance, comparing embedding spaces generated from different linguistic communities, where the same legal concepts are embedded in distinct traditions of usage. If the instrument detects stable differences under such conditions, it possesses discriminating power; if it does not, the measured structures may be artifacts of the model rather than properties of the legal system.

The hierarchy of these three questions corresponds to what Edelman, in a different context, described as a hierarchy of representational fidelity: from the minimal condition of distinction-preservation (the embedding space at least keeps legal concepts apart from non-legal noise) through nearest-neighbour preservation (local conceptual neighbourhoods are stable) to the full preservation of the similarity spectrum (the complete relational structure is maintained across models and traditions). The experimental programme of this thesis, developed in Chapters 2 and 3, ascends this hierarchy step by step.[^79]

Aceves and Evans have captured the methodological posture with an apt analogy: embedding models, they argue, “must be subject to structured measurements (like a psychometric questionnaire offered to human subjects) to render their conceptual landscape interpretable.” The embedding space does not speak for itself. It requires a battery of structured probes, each designed to elicit a specific dimension of its internal organisation, and each subjected to statistical tests that distinguish signal from noise. RSA, neighbourhood analysis, and axis projection are, in this framework, the structured measurements that transform a high-dimensional geometric object into interpretable claims about legal meaning.[^80]

The measurability thesis has been stated, and the instrument validation logic that follows from it has been outlined. But the claim requires delimitation. The measurability thesis is powerful precisely because it is limited. The next section specifies, with equal care, what this thesis does not claim.



---

[^62]: The most common alignment strategies include shared subword vocabularies (as in multilingual BERT), translation language modelling with parallel corpora (as in XLM), and distillation from a teacher model trained on translation pairs. In each case, the alignment is a training objective, not an empirical finding. See CONNEAU, A. and LAMPLE, G., “Cross-lingual Language Model Pretraining,” in Advances in Neural Information Processing Systems 32 (NeurIPS 2019), 2019, pp. 7059-7069.


[^63]: SHEPARD, R.N. and CHIPMAN, S., “Second-order isomorphism of internal representations: Shapes of states,” Cognitive Psychology, 1(1), 1970, pp. 1-17.


[^64]: Ibid., p. 2. The formulation has become canonical: Edelman (1998) quotes it verbatim (p. 450), and Kriegeskorte, Mur, and Bandettini (2008) cite it as the conceptual foundation of RSA.


[^65]: Ibid., p. 2. The example is pedagogically effective because it crosses sensory modalities: the “taste of persimmon” is maximally distant from the shape of a square not on any single dimension, but across the entire representational space. The distance is relational, not dimensional.


[^66]: EDELMAN, S., “Representation is representation of similarities,” Behavioral and Brain Sciences, 21(4), 1998, pp. 449-498, at p. 451. Edelman’s formulation generalises Shepard’s principle from the specific case of state shapes to a theory of neural representation as such. His argument is that the brain does not build internal models of objects; it builds a space of similarities among objects, and this space is the representation.


[^67]: KRIEGESKORTE, N., MUR, M. and BANDETTINI, P., “Representational similarity analysis: connecting the branches of systems neuroscience,” Frontiers in Systems Neuroscience, 2(4), 2008, pp. 1-28. The paper proposed RSA as a framework for comparing representations across brain regions (measured by fMRI), species (human vs. monkey), and computational models (neural networks, feature-based models). The ambition was explicitly integrative: RSA would provide a “common language” for the fragmented branches of systems neuroscience.


[^68]: Ibid., p. 4. The sentence confirms the direct intellectual lineage: RSA is the operationalisation of second-order isomorphism.


[^69]: NILI, H., WINGFIELD, C., WALTHER, A., SU, L., MARSLEN-WILSON, W. and KRIEGESKORTE, N., “A toolbox for representational similarity analysis,” PLoS Computational Biology, 10(4), 2014, e1003553, at p. 2. Nili et al. codified the RSA methodology into a standardised toolbox with recommended statistical procedures.


[^70]: KRIEGESKORTE, MUR, and BANDETTINI (2008), p. 4. The phrase “abstract from the spatial layout” is the key: dimensionality, axis orientation, and absolute position are discarded. Only the relational structure survives.


[^71]: Ibid., pp. 10-11: “For the models we use here, we do not wish to assume a linear match between dissimilarity matrices. We therefore use the Spearman rank correlation coefficient to compare them.” The rank transformation ensures that the comparison is sensitive to the ordering of distances, not their absolute scale, a property that is essential when comparing spaces produced by different model architectures with different distance scales.


[^72]: Ibid., p. 11: “We therefore suggest testing the relatedness of dissimilarity matrices by randomizing the condition labels. We choose a random permutation of the conditions, reorder rows and columns of one of the two dissimilarity matrices to be compared according to this permutation, and compute the correlation. Repeating this step many times (e.g., 10,000 times), we obtain a distribution of correlations simulating the null hypothesis that the two dissimilarity matrices are unrelated.” This procedure is equivalent to the Mantel test (MANTEL, N., “The detection of disease clustering and a generalized regression approach,” Cancer Research, 27(2), 1967, pp. 209-220), adapted from spatial statistics to representational geometry.


[^73]: DIEDRICHSEN, J. and KRIEGESKORTE, N., “Representational models: A common framework for understanding encoding, pattern-component, and representational-similarity analysis,” PLoS Computational Biology, 13(4), 2017, e1005508, at p. 3.


[^74]: Ibid. Diedrichsen and Kriegeskorte further showed that three previously distinct analysis methods (encoding analysis, pattern-component modelling, and RSA) all evaluate the same mathematical object: the second moment of the distribution of activity profiles. The convergence of methods onto a single underlying quantity reinforces the claim that the RDM captures something fundamental about representational structure, not merely one perspective among many.


[^75]: SHEPARD and CHIPMAN (1970), p. 17. The passage continues with a quotation from Garner: “the factors known in perception are properties of sets of stimuli, not properties of individual stimuli” (GARNER, W. R., “To perceive is to know,” American Psychologist, 21(1), 1966, p. 11). The principle applies with equal force to legal concepts in embedding spaces: the meaningful information is in the set of relations, not in any single vector.


[^76]: ACEVES, P. and EVANS, J.A., “Mobilizing conceptual spaces: How word embedding models can inform measurement and theory within organization science,” Organization Science, Articles in Advance, 2023, pp. 1-27, at p. 2.


[^77]: The resonance with Savigny’s Volksgeist doctrine (1.2) is not accidental. If law, like language, emerges from the collective consciousness of a people (SAVIGNY, System cit., 7), then a computational model trained on that people’s linguistic output is, in a precise sense, a statistical sediment of its Volksgeist. One could advance the stronger, ontological claim that the model is a crystallisation of that collective consciousness. The present thesis makes the weaker, epistemological claim: the embedding space is not the Volksgeist itself but an instrument through which certain properties of its semantic organisation can be observed and measured.


[^78]: BHUPATIRAJU, S., CHEN, D.L. and VENKATARAMANAN, K., “Mapping the geometry of law using natural language processing,” European Journal of Empirical Legal Studies, 1(1), 2024, pp. 49-68, at p. 49. The title of their paper, “Mapping the geometry of law,” is itself indicative: the spatial metaphor has entered the legal NLP literature. What remains missing, and what the present thesis provides, is the epistemological justification for treating this geometry as a legitimate object of legal knowledge rather than merely a convenient computational representation.


[^79]: EDELMAN (1998), p. 452. The hierarchy distinguishes: (1) distinction-preserving representations (which merely keep stimuli apart), (2) nearest-neighbour preserving representations (which maintain local structure), and (3) similarity-spectrum preserving representations (which maintain the full pattern of pairwise relations). The experimental chapters of this thesis test progressively higher levels of this hierarchy.


[^80]: ACEVES and EVANS (2023), p. 6. The full passage draws an explicit parallel with psychometric methodology: “In the same way that structured psychological and anthropological elicitations assume that human conceptual understandings are not sufficiently interpretable and comparable ‘on their own,’ so do we argue that embedding models must be subject to structured measurements (like a psychometric questionnaire offered to human subjects) to render their conceptual landscape interpretable.” The analogy is apt: the embedding space, like the human mind, does not yield its structure to casual inspection. It requires systematic probing.

