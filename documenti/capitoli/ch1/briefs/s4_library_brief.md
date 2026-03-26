# Library Brief: Local PDF Sources for Section 1.4
**Section**: 1.4 — Space as legal instrument: second-order isomorphism and the measurability thesis
**Date**: 2026-03-08
**Prepared by**: Librarian agent (from local PDF sources)

---

## 1. SHEPARD & CHIPMAN (1970)
**PDF**: `shepard1970.pdf` (also: `Second-order isomorphism of internal representations_...Anna's Archive.pdf`)
**Full ref**: Shepard, R.N. & Chipman, S. (1970). Second-order isomorphism of internal representations: Shapes of states. *Cognitive Psychology*, 1(1), 1--17. doi:10.1016/0010-0285(70)90002-2

### Quote 1 — Abstract: definition of second-order isomorphism (p. 1) [VERIFIED]
> "It is argued that, while there is no structural resemblance between an individual internal representation and its corresponding external object, an approximate parallelism should nevertheless hold between the relations among different internal representations and the relations among their corresponding external objects."

**Relevance**: This is the foundational definition of second-order isomorphism. For 1.4.b, it provides the core conceptual move: we do not need representations to *resemble* objects; we need them to preserve *relations among* objects. This is exactly the logic that justifies comparing embedding spaces via their relational structure (RDMs) rather than via direct vector correspondence.

### Quote 2 — The proposed concept (p. 2) [VERIFIED]
> "The crucial step consists in accepting that the isomorphism should be sought---not in the first-order relation between (*a*) an individual object, and (*b*) its corresponding internal representation---but in the second-order relation between (*a*) the relations among alternative external objects, and (*b*) the relations among their corresponding internal representations."

**Relevance**: The single most important quote for 1.4.b. This is the sentence that Edelman (1998), Kriegeskorte et al. (2008), and the entire RSA tradition build upon. For the thesis, this justifies the methodological move from "what does a single embedding look like?" to "what does the pattern of distances among embeddings look like?"

### Quote 3 — Functional relation and subjective similarity (p. 2) [VERIFIED]
> "Thus, although the internal representation for a square need not itself be square, it should (whatever it is) at least have a closer functional relation to the internal representation for a rectangle than to that, say, for a green flash or the taste of persimmon."

**Relevance**: Provides the intuitive example for 1.4.b. Translatable directly into the legal domain: the embedding of "due process" need not look like due process, but it should be closer to "fair trial" than to "eminent domain." This is the bridge sentence between cognitive psychology and the thesis's legal application.

### Quote 4 — Discussion and Conclusions: what similarity data tell us (p. 17) [VERIFIED]
> "What they can, however, tell us about is the relations between that internal representation and other internal representations. In this sense, then, we would extend to memory and imagination what Garner (1966, p. 11) has already asserted with regard to direct perception; namely, that 'the factors known in perception are properties of sets of stimuli, not properties of individual stimuli.'"

**Relevance**: Key for 1.4.d (measurability thesis). The insight that meaningful information lies in *relations among representations*, not in individual representations, is the epistemological foundation for the claim that relational structure in embedding spaces constitutes a measurable object. Garner's dictum ("properties of sets, not of individuals") applies directly to legal concepts in embedding spaces.

### Quote 5 — Concluding evidence claim (p. 17) [VERIFIED]
> "We hope that the present study at least demonstrates the possibility of investigating the structure of internal representations at this more abstract level and, at the same time, provides evidence at this level for a kind of 'second-order' isomorphism between internal representations and their corresponding external objects."

**Relevance**: Shepard & Chipman's self-assessment of their contribution. For the thesis, this frames the ambition: investigating the *structure* of legal representations at the relational level, providing evidence for a second-order isomorphism between legal concepts (as culturally understood) and their embedding representations.

---

## 2. KRIEGESKORTE, MUR & BANDETTINI (2008)
**PDF**: `frontiers_kriegeskorte2008_rsa.pdf`
**Full ref**: Kriegeskorte, N., Mur, M. & Bandettini, P. (2008). Representational similarity analysis --- connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience*, 2(4), 1--28. doi:10.3389/neuro.06.004.2008

### Quote 1 — Core idea: RDM as signature (p. 4) [VERIFIED]
> "The RDMs serve as the signatures of regional representations in brains and models. Importantly, these signatures abstract from the spatial layout of the representations. They are indexed (horizontally and vertically) by experimental condition and can thus be directly compared between brain and model. What we are comparing, intuitively, is the represented information, not the activity patterns themselves."

**Relevance**: Central for 1.4.c. The RDM abstracts away from the "spatial layout" (i.e., the specific dimensionality and axis alignment of the space). This is exactly why RDMs solve the cross-space comparison problem described in 1.4.a: a 768-dim English space and a 1024-dim Chinese space cannot be directly compared, but their RDMs can.

### Quote 2 — Second-order isomorphism as the core concept (p. 4) [VERIFIED]
> "One concept at the core of our approach is that of second-order isomorphism (Shepard and Chipman, 1970), i.e., the match of dissimilarity matrices."

**Relevance**: Establishes the direct intellectual lineage from Shepard & Chipman (1970) to RSA. For 1.4.c, this confirms that RSA is the operationalization of second-order isomorphism.

### Quote 3 — Randomization test for RDM relatedness (p. 11) [VERIFIED]
> "We therefore suggest testing the relatedness of dissimilarity matrices by randomizing the condition labels. We choose a random permutation of the conditions, reorder rows and columns of one of the two dissimilarity matrices to be compared according to this permutation, and compute the correlation. Repeating this step many times (e.g., 10,000 times), we obtain a distribution of correlations simulating the null hypothesis that the two dissimilarity matrices are unrelated."

**Relevance**: Key for 1.4.c (statistical inference). This is the Mantel test adapted for RDMs. The permutation approach is nonparametric and makes no distributional assumptions, which is important for the thesis's methodological rigor. Connects to 1.4.e (instrument validation): if the permutation test rejects the null, we have evidence that the instrument detects a signal.

### Quote 4 — Comparing brain and model dissimilarity matrices (p. 10--11) [VERIFIED]
> "One way to quantify the match between two dissimilarity matrices is by means of a correlation coefficient. We use 1-correlation as a measure of the dissimilarity between RDMs [...]. For the models we use here, we do not wish to assume a linear match between dissimilarity matrices. We therefore use the Spearman rank correlation coefficient to compare them."

**Relevance**: For 1.4.c, this specifies the operational measure: Spearman rank correlation between the upper triangles of two RDMs. The choice of rank correlation (rather than Pearson) avoids linearity assumptions, which is methodologically important when comparing representations from fundamentally different systems (brains vs. models; English embeddings vs. Chinese embeddings).

---

## 3. EDELMAN (1998)
**PDF**: `edelman1998_representation_similarities_bbs.pdf`
**Full ref**: Edelman, S. (1998). Representation is representation of similarities. *Behavioral and Brain Sciences*, 21(4), 449--498. doi:10.1017/S0140525X98001253

### Quote 1 — Title thesis (p. 449, abstract) [VERIFIED]
> "This shape space supports representations of distal shape similarities that are veridical as Shepard's (1968) second-order isomorphisms (i.e., correspondence between distal and proximal similarities among shapes, rather than between distal shapes and their proximal representations)."

**Relevance**: For 1.4.b, Edelman extends Shepard's insight into a full theory of representation. The formulation "correspondence between distal and proximal similarities" is a more precise restatement of second-order isomorphism that can be adapted for legal concepts: correspondence between conceptual similarities (as culturally understood) and embedding similarities (as geometrically measured).

### Quote 2 — The alternative answer: represent similarities, not shapes (p. 451) [VERIFIED]
> "The approach expounded below, which is closely related to Shepard's (1968) idea of representation by second-order isomorphism, offers such an alternative answer: *represent similarity between shapes, not the geometry of each shape in itself.*"

**Relevance**: For 1.4.b and 1.4.d. This is the conceptual pivot: what matters is not the geometry of each individual representation but the *similarities between* representations. Translated into the legal domain: what matters is not the 768 coordinates of a single legal concept's embedding but the pattern of distances among all legal concepts. This directly supports the measurability thesis.

### Quote 3 — Quoting Shepard & Chipman on the crucial step (p. 450) [VERIFIED]
> "Quoting Shepard and Chipman (1970, p. 2), 'the isomorphism should be sought -- not in the first-order relation between (*a*) an individual object, and (*b*) its corresponding internal representation -- but in the second-order relation between (*a*) the relations among alternative external objects, and (*b*) the relations among their corresponding internal representations.'"

**Relevance**: Edelman's re-quotation of the Shepard & Chipman passage confirms its canonical status. Having two independent sources quoting the same passage strengthens its weight in the thesis. This is also useful because Edelman's framing (1998) is closer in time to the computational tradition the thesis draws on.

### Quote 4 — Levels of representation fidelity (p. 452) [VERIFIED]
> "Whereas the lowest-fidelity (distinction-preserving) representation does not necessarily preserve such properties, the highest-fidelity (similarity-preserving) representation clearly does."

**Relevance**: For 1.4.d and 1.4.e. Edelman distinguishes levels of representational fidelity: (1) distinctness-preserving (merely keeping things apart), (2) nearest-neighbor preserving, (3) full similarity spectrum preservation. This hierarchy maps onto the thesis's instrument validation logic: does the embedding space merely distinguish legal concepts (weak), preserve local neighborhoods (medium), or preserve the full relational structure (strong)?

---

## 4. NILI ET AL. (2014)
**PDF**: `plos_nili2014_rsa_toolbox.pdf`
**Full ref**: Nili, H., Wingfield, C., Walther, A., Su, L., Marslen-Wilson, W. & Kriegeskorte, N. (2014). A toolbox for representational similarity analysis. *PLoS Computational Biology*, 10(4), e1003553. doi:10.1371/journal.pcbi.1003553

### Quote 1 — RDM as signature of representational geometry (p. 2) [VERIFIED]
> "RSA characterizes the representation in each brain region by a representational dissimilarity matrix (RDM, Figure 1). The most basic type of RDM is a square symmetric matrix, indexed by the stimuli horizontally and vertically (in the same order). The diagonal entries reflect comparisons between identical stimuli and are 0, by definition, in this type of RDM. Each off-diagonal value indicates the dissimilarity between the activity patterns associated with two different stimuli. The dissimilarities can be interpreted as distances between points in the multivariate response space. The RDM thus describes the geometry of the arrangement of patterns in this space."

**Relevance**: For 1.4.c. This is the clearest operational definition of the RDM in the literature. The final sentence ("the RDM thus describes the geometry of the arrangement of patterns in this space") is directly transferable: the RDM describes the geometry of the arrangement of legal concepts in embedding space.

### Quote 2 — Statistical inference: stimulus-label randomization test (p. 8) [VERIFIED]
> "The toolbox uses frequentist nonparametric inference procedures. For testing the relatedness of two RDMs, the preferred (and default) method is the signed-rank test across subjects. [...] The fixed-effects alternative is to test RDM relatedness using the stimulus-label randomization test. This test is definitely valid and expected to be more powerful than the signed-rank test across subjects, because it tests a less ambitious hypothesis: that the RDMs are related in the experimental group of subjects, rather than in the population. The stimulus-label randomization test can be used for a single subject or a group of subjects of any size."

**Relevance**: For 1.4.c and 1.4.e. Nili et al. provide the statistical framework for testing whether the similarity structure in one space is related to that in another. The stimulus-label randomization test (i.e., the Mantel test) is the default when comparing RDMs from different systems without repeated measures. This is exactly the thesis's situation: two embedding spaces (WEIRD, Sinic) without repeated measures.

### Quote 3 — Bootstrap for condition set (p. 8) [VERIFIED]
> "The relatedness of two RDMs can also be tested by bootstrapping the stimulus set and/or the subjects set. The motivation for bootstrapping is to simulate repeated sampling from the population. Bootstrapping, thus, can help generalize from the sampled subjects and/or stimuli to the population of subjects and/or the population of stimuli."

**Relevance**: For 1.4.c and 1.4.e. Bootstrap resampling of the stimulus set (= legal terms, in the thesis) provides confidence intervals and generalizes findings from the specific term set to the broader population of legal concepts. This is the block bootstrap approach used in the thesis pipeline.

---

## 5. DIEDRICHSEN & KRIEGESKORTE (2017)
**PDF**: `plos_diedrichsen2017_representational_models.pdf`
**Full ref**: Diedrichsen, J. & Kriegeskorte, N. (2017). Representational models: A common framework for understanding encoding, pattern-component, and representational-similarity analysis. *PLoS Computational Biology*, 13(4), e1005508. doi:10.1371/journal.pcbi.1005508

### Quote 1 — The second moment defines representational geometry (p. 3) [VERIFIED]
> "In this paper, we show that the multivariate *second moment* of the activity profiles fully defines the representational geometry and with it all the information that can linearly or nonlinearly decoded. In particular, under the assumption of Gaussian noise the second moment determines the signal-to-noise ratio with which any feature can be decoded."

**Relevance**: For 1.4.c and 1.4.d. Diedrichsen & Kriegeskorte formalize the intuition behind RSA: the second moment (= pairwise distances/dissimilarities) is not just a convenient summary but a *complete* characterization of the representational geometry. This strengthens the measurability thesis: the RDM is not a lossy compression of the embedding space; it captures all the geometrically available information about relational structure.

### Quote 2 — Representational models as hypotheses (p. 3) [VERIFIED]
> "Representational models, as considered here, go one step further: they fully characterize the representational geometry, defining all represented features in a region, how strongly each of them is represented (signal to noise ratio), and how the activity patterns associated with different features relate to each other. Representational models therefore fully specify the representational content of an area."

**Relevance**: For 1.4.e (instrument validation). The representational model framework means that comparing an observed RDM to a predicted RDM is a hypothesis test about the content of a representation. In the thesis, this translates to: comparing the RDM from an English embedding space to the RDM from a Chinese embedding space tests the hypothesis that the two systems organize legal concepts similarly.

### Quote 3 — Three methods share one core (p. 1, abstract) [VERIFIED]
> "Here we develop a common mathematical framework for understanding the relationship of these three methods, which share one core commonality: all three evaluate the second moment of the distribution of activity profiles, which determines the representational geometry, and thus how well any feature can be decoded from population activity."

**Relevance**: For 1.4.c. Confirms that the "second moment" (i.e., the pairwise distance structure) is the fundamental object of analysis across multiple methods. This provides mathematical grounding for why the thesis focuses on distance matrices: they are the natural and sufficient object for characterizing representational geometry.

---

## 6. BHUPATIRAJU, CHEN & VENKATARAMANAN (2024) [OPTIONAL]
**PDF**: `ejels_bhupatiraju2024_mapping_geometry_law_nlp.pdf`
**Full ref**: Bhupatiraju, S., Chen, D.L. & Venkataramanan, K. (2024). Mapping the geometry of law using natural language processing. *European Journal of Empirical Legal Studies*, 1(1), 49--68. doi:10.62355/ejels.18073

### Quote 1 — Law is embedded in language (p. 49) [VERIFIED]
> "Law is embedded in language. In this paper, we ask what can be gained by applying new techniques from natural language processing (NLP) to the law. NLP translates words and documents into vectors within a high-dimensional Euclidean space. Vector representations of words and documents are information-dense in the sense of retaining information about semantic content and meaning, while also being computationally tractable."

**Relevance**: For 1.4.d. Direct precedent: Bhupatiraju et al. explicitly treat legal texts as geometric objects in embedding space. Their framing ("mapping the geometry of law") validates the thesis's central metaphor. However, their approach is applied (document embeddings for prediction), not epistemological (measuring meaning structure). The thesis goes further by providing the theoretical framework (second-order isomorphism) that justifies the geometric approach.

### Quote 2 — Geometric relations between legal sources (p. 50) [VERIFIED]
> "This new approach to legal studies addresses shortcomings of existing methods for studying legal language. Because law consists of text, research methods based on formal math and numerical data are limited by the questions that can be asked. The formal theory literature has approached the law metaphorically. This case-space literature, in particular, treats the law spatially, where the law separates the fact space into 'liable' and 'not liable' and 'not guilty' and 'guilty'. Case-space models give us some intuition into the legal reasoning process. But they have been somewhat limited empirically because it has been infeasible to measure the legal case space."

**Relevance**: For 1.4.d. Bhupatiraju et al. identify the gap: formal "case-space" models were metaphorical and unmeasurable. Embedding spaces make the geometric metaphor empirically tractable. This validates the thesis's claim that embedding geometry provides a measurable object for legal science.

---

## 7. ACEVES & EVANS (2023) [OPTIONAL]
**PDF**: `socarxiv_aceves2023_mobilizing_conceptual_spaces_orgsci.pdf`
**Full ref**: Aceves, P. & Evans, J.A. (2023). Mobilizing conceptual spaces: How word embedding models can inform measurement and theory within organization science. *Organization Science*, Articles in Advance, 1--27. doi:10.1287/orsc.2023.1686

### Quote 1 — Central claim: embeddings represent conceptual spaces (p. 2) [VERIFIED]
> "Our primary argument here is that each vector within a word embedding model represents a concept and that the entire embedding model represents the conceptual space of the social system that generated the textual data. Conceptual spaces as represented by embedding models are multidimensional spaces within which concepts ranging from norms and knowledge to ideas and inventions relate to one another."

**Relevance**: For 1.4.d. Aceves & Evans articulate for organizational science what the thesis articulates for legal science: the embedding model represents the conceptual space of a social system. The parallel is direct: a legal corpus generates an embedding space that represents the conceptual space of a legal tradition.

### Quote 2 — Embeddings as structured psychometric instruments (p. 6) [VERIFIED]
> "In what follows, we propose a range of prompts and measurements on these conceptual spaces that function as methods of producing structured interpretations of them. This is much like psychologists use psychometric surveys to turn conceptual impressions into interpretable opinions, person by person [...]. In the same way that structured psychological and anthropological elicitations assume that human conceptual understandings are not sufficiently interpretable and comparable 'on their own,' so do we argue that embedding models must be subject to structured measurements (like a psychometric questionnaire offered to human subjects) to render their conceptual landscape interpretable."

**Relevance**: For 1.4.d and 1.4.e. This is the methodological analogy that bridges cognitive science and the thesis: embedding spaces are not self-interpreting; they require structured measurement (RSA, neighborhood analysis, axis projection) to yield interpretable results. The psychometric analogy is powerful: just as you need a validated instrument to measure personality traits, you need validated geometric probes to measure legal meaning structure.

### Quote 3 — Word embeddings as "digital double" of collective knowledge (p. 2) [VERIFIED]
> "For the purpose of organization science, these embedding models create a 'digital double' of the collective knowledge held by individuals within a social system."

**Relevance**: For 1.4.d. The "digital double" metaphor parallels the thesis's concept of the embedding space as a measurable proxy for the culturally constructed meaning of legal concepts. This also resonates with the DigiVolksgeist paper (Puzio), which is the thesis's intellectual antecedent.

---

## CROSS-REFERENCE MATRIX

| Source | 1.4.a (cross-space problem) | 1.4.b (second-order iso.) | 1.4.c (RSA/RDM) | 1.4.d (measurability thesis) | 1.4.e (validation logic) |
|---|---|---|---|---|---|
| Shepard & Chipman 1970 | | Q1, Q2, Q3 (core) | | Q4, Q5 | |
| Kriegeskorte et al. 2008 | Q1 (abstracts from layout) | Q2 (cites Shepard) | Q3, Q4 (core) | | Q3 (permutation test) |
| Edelman 1998 | | Q1, Q2, Q3 (extends Shepard) | | Q2 (represent similarities) | Q4 (fidelity hierarchy) |
| Nili et al. 2014 | | | Q1 (RDM def), Q2, Q3 (core) | | Q2, Q3 (statistical tests) |
| Diedrichsen & Kriegeskorte 2017 | | | Q1, Q3 (second moment) | Q1 (completeness) | Q2 (hypothesis testing) |
| Bhupatiraju et al. 2024 | | | | Q1, Q2 (legal precedent) | |
| Aceves & Evans 2023 | | | | Q1, Q3 (conceptual spaces) | Q2 (structured measurement) |

---

## SUGGESTED ARGUMENT FLOW FOR 1.4

1. **Open with the problem** (1.4.a): Two embedding spaces with different dimensionalities (768 vs. 1024) and different training regimes cannot be directly compared. Multilingual models impose rather than discover correspondence. Therefore: monolingual models + a comparison method that abstracts from dimensionality.

2. **Introduce second-order isomorphism** (1.4.b): Shepard & Chipman (1970) showed that meaningful information resides not in individual representations but in the *relations among* representations. Use Quotes 2 and 3 from Shepard. Reinforce with Edelman (1998), Quote 2: "represent similarity between shapes, not the geometry of each shape in itself."

3. **Operationalize via RSA** (1.4.c): Kriegeskorte et al. (2008) turned second-order isomorphism into a quantitative framework. The RDM captures the complete relational structure (Diedrichsen & Kriegeskorte 2017, Quote 1). Comparing RDMs via Spearman rho + permutation test (Kriegeskorte Quote 3; Nili et al. Quote 2) provides a statistically grounded comparison that works across spaces of any dimensionality.

4. **State the measurability thesis** (1.4.d): Given the above, the relational organization of legal concepts in embedding spaces constitutes an observable, measurable, and empirically testable object. Support with Shepard Quote 4 (relations are what we can measure), Aceves & Evans Quote 1 (embedding = conceptual space), Bhupatiraju et al. Quote 2 (making legal geometry empirically tractable).

5. **Instrument validation logic** (1.4.e): Three questions: (i) Does a signal exist? (permutation test rejects null), (ii) Is the instrument reliable? (cross-model consistency), (iii) Does it discriminate? (different legal traditions produce detectably different structures). Support with Edelman Quote 4 (fidelity hierarchy), Nili et al. Quotes 2--3 (statistical procedures), Aceves & Evans Quote 2 (structured measurement analogy).
