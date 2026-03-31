# Chapter 1 — Meaning without measure

## 1.5 What this work does not claim


The argument of this chapter has moved through four stages: from the identification of an epistemological gap in legal science (1.1), through the philosophical tradition that constitutes legal meaning as collective practice (1.2), to the computational formalisation of that practice as geometric structure (1.3), and finally to the framework of second-order isomorphism that renders such structures comparable and measurable (1.4). The measurability thesis has been stated. Before proceeding to the construction of the apparatus, it is necessary to state, with equal precision, what the thesis does not claim.


### Law is not geometry


The present work does not propose that law is geometry, that normative phenomena can be reduced to spatial relations, or that the richness of legal reasoning can be captured in a matrix of pairwise distances. The claim is narrower and more specific: that geometry can serve as a legitimate instrument of legal knowledge, one that renders visible a dimension of meaning that has hitherto been accessible only through hermeneutic reconstruction.

The distinction between instrument and object is fundamental. The telescope analogy introduced in 1.1 bears repeating in its negative formulation: the lenses did not assert that celestial bodies are optical phenomena; they used optics to observe them. In the same way, the embedding space does not assert that legal meaning is geometric. It uses geometry to observe a dimension of meaning, the relational structure among concepts, that would otherwise remain accessible only through hermeneutic reconstruction. The instrument is interposed between the jurist and the semantic organisation of a legal tradition. It does not claim that law is reducible to its output, any more than the telescope claimed that Jupiter’s moons were reducible to the refraction of light.


### What embeddings capture, and what they do not


It is equally essential to be precise about the object that the instrument observes. What embedding spaces capture is distributional relational structure: the statistical sediment of how a linguistic community uses its legal concepts, which terms appear in similar contexts, which are kept apart, which cluster together into recognisable domains. This is a genuine dimension of meaning. It is not the whole of it.

What embeddings do not capture is substantial. They do not capture argumentation: the chain of reasons that a court marshals in support of a conclusion. They do not capture institutional context: the fact that a constitutional provision occupies a different normative rank than a ministerial regulation, even if both employ the same vocabulary. They do not capture legislative intent, normative force, or precedential reasoning. They do not capture the hermeneutic richness of interpretation, the layered interplay of text, context, purpose, and tradition that Betti’s canons sought to discipline and that Gadamer’s wirkungsgeschichtliches Bewusstsein sought to describe. The object of measurement in this thesis is the structure of meaning, the geometry of relations between concepts, not meaning in its full juristic sense. The distinction is not a concession wrung from the argument; it is built into the argument’s foundation. As the measurability thesis (1.4) makes explicit, what is claimed to be observable and measurable is the relational organisation of concepts, not the totality of what law means.


### Measurement does not replace interpretation


The relationship between measurement and interpretation is complementary, not competitive. The embedding space is the telescope, not the astronomer. The instrument can reveal that two legal traditions organise their constitutional vocabulary in geometrically distinct ways, that certain concepts occupy isolated positions in one tradition and central positions in another, that a domain which is tightly cohesive in one legal culture is fragmented in another. These are observations. They become legal knowledge only when a jurist, equipped with doctrinal understanding and comparative sensibility, interprets them: asks why the divergence exists, what institutional or historical factors might account for it, and what consequences it carries for the translatability of legal norms across systems.

The instrument provides the measure; the jurist provides the judgment. This division of labour is not a limitation but a feature. It is precisely because the instrument does not interpret that its output is intersubjectively accessible: two jurists examining the same representational dissimilarity matrix will observe the same distances, even if they assign different legal significance to what they observe. The interpretive task, far from being eliminated, is sharpened: the jurist no longer debates whether a divergence exists (the instrument settles this) but what the divergence means. Chapter 4, under the heading Quod numeri tacent (what the numbers do not say), returns to this boundary in detail.[^81]


### Models are not culturally neutral


No embedding model is a view from nowhere. Every model is trained on a corpus produced by a specific linguistic community, and the distributional patterns it absorbs are the patterns of that community’s practices. Each model, to borrow the language of the Historical School, has its own Volk. The models employed in this thesis have each absorbed the semantic structure of the corpus on which they were trained; a model trained on anglophone texts and a model trained on sinophone texts will reflect, respectively, the distributional practices of their own linguistic communities. The resulting embedding spaces are not neutral coordinate systems; they are cultural artifacts, shaped by the practices they encode.[^82]

The methodological design of the thesis addresses this condition without pretending to eliminate it. The monolingual architecture, in which each tradition is represented by models trained exclusively on its own linguistic community’s texts, isolates the cultural variable: any systematic difference between the resulting embedding spaces reflects a difference in the distributional practices of the two communities, not an artifact of cross-lingual alignment. But isolation is not elimination. The models remain cultural objects, and their outputs must be read as such. The panel of three independently trained models per tradition serves as a robustness control: a finding is attributed to the tradition, rather than to the idiosyncrasies of a single model, only if it replicates across the panel.[^83]


### Correlation, not causation


The geometric patterns that the apparatus reveals are structural correlations, not causal explanations. If the embedding spaces of the English-language and Chinese-language models organise a particular legal domain differently, the thesis can describe the difference, quantify it, and assess its statistical reliability. It cannot, from geometry alone, explain why the difference exists. The causes may be historical (centuries of divergent institutional development), linguistic (the categorial structure of the language itself), political (the influence of particular legislative traditions), or some combination of all three. Geometric measurement identifies the quod (what the structure is); it does not pronounce upon the cur (why the structure is so). The inferential discipline adopted throughout this work (2.4) enforces this boundary rigorously: every empirical observation is accompanied by an explicit statement of what the measurement can and cannot support.[^84]


### This is not comparative law


Finally, and most consequentially for the disciplinary positioning of the thesis: the cross-tradition comparison that structures the empirical chapters is the experimental design, not the research question. The thesis does not ask how different legal traditions diverge. It asks whether embedding spaces can serve as instruments of legal knowledge, and if so, with what reliability and within what limits.

A measurement instrument cannot be validated in a vacuum. It requires variation: known or expected differences against which the instrument’s sensitivity can be tested. A thermometer is validated by exposing it to substances at different temperatures; a telescope is validated by directing it toward objects at different distances. The cross-tradition comparison serves the same function. The specific experimental setting, in which the same legal concepts exist in two linguistic traditions within a shared normative framework, furnishes the variation necessary to test whether the instrument detects differences where institutional and linguistic conditions predict them. If the instrument detects no structure, it fails the first test (signal). If it detects structure but that structure is unstable across models, it fails the second (reliability). If it detects stable structure but cannot discriminate between conditions where differences are expected and conditions where they are not, it fails the third (discriminating power). These are questions of methodology, not of comparative law. The answers may, in time, benefit the comparatist. But they belong, in the first instance, to the epistemology and methodology of legal science.[^85]


### From argument to apparatus


The theoretical argument is now complete. Chapter 1 has established that legal science operates upon meaning without possessing an instrument to measure it (1.1); that the philosophical tradition from Savigny to Wittgenstein constitutes legal meaning as a collective, practice-based, and in principle observable phenomenon (1.2); that the distributional hypothesis and its computational implementations transform this philosophical insight into geometric representations amenable to systematic analysis (1.3); and that the framework of second-order isomorphism provides the method for comparing such representations across independently constructed spaces (1.4). The present section has circumscribed the claim: the thesis proposes an instrument, not a reduction; it measures structure, not the totality of meaning; it describes correlations, not causes; and its comparative design is a validation protocol, not a contribution to comparative law.

What remains is to build the instrument. Chapter 2 turns from the question of whether legal meaning can be measured to the question of how: the construction of the dataset, the selection and justification of the embedding models, the statistical apparatus, and the experimental design through which the measurability thesis is put to empirical test.



---

[^81]: The phrase quod numeri tacent is adapted, with deliberate liberty, from the classical quod non est in actis non est in mundo. Where the procedural maxim warns that what is not in the record does not exist for the court, the adapted form warns that what the numbers do not say does not exist for the instrument. The jurist’s task begins precisely where the number falls silent.


[^82]: The formulation transposes Savigny’s insight (1.2) into a computational key. If law has its root in the common consciousness of a people (System cit., 7), and if language is the medium through which that consciousness expresses itself (Humboldt, Über die Verschiedenheit cit., 9), then a model trained exclusively on a given linguistic community’s texts absorbs, involuntarily, the semantic structure of that community’s conceptual world. The present thesis treats this absorption epistemologically: the model is not the Volksgeist itself but an instrument through which certain properties of the Volksgeist become observable.


[^83]: The panel design is discussed in detail in 2.3. Multiple independently trained models per tradition serve as a replication mechanism: a result that appears with one model but not with others is treated as model-specific; a result that replicates across the majority of models is attributed to the tradition.


[^84]: The distinction between quod and cur echoes, in a different key, the Aristotelian distinction between knowledge quia (that something is the case) and knowledge propter quid (why it is the case). See ARISTOTLE, Posterior Analytics, I.13. Geometric measurement of embedding spaces yields knowledge of the first kind; causal explanation requires the second, and lies beyond the instrument’s reach.


[^85]: The experimental setting and the rationale for its selection are discussed in 2.1. The key desideratum is a legal environment in which the same normative concepts are expressed in two linguistic traditions within a single institutional framework, so that the institutional variable is controlled and the linguistic-cultural variable is isolated.

