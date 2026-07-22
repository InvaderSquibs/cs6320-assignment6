# Assignment 6: Tabular Modeling and Model-Choice Justification

- **Due** Jun 30 by 11:59pm
- **Points** 100
- **Submitting** a file upload
- **File Types** zip

# CS 6320 Assignment 6: Tabular Modeling and Model-Choice Justification

## Purpose

Many real client problems involve structured or tabular data, and many non-tabular problems can still be inspected through structured proxy features. This assignment asks you to use your portfolio problem as the context for model choice: is a tabular representation, a tabular baseline, or a neural tabular model justified for your portfolio dataset?

This week continues the evaluation discipline from Assignment 5. The goal is not to prove that a neural model is best. The goal is to make a fair model-choice argument for your portfolio project using shared data preparation, metrics, evidence, and practical constraints such as interpretability, cost, maintainability, and responsible use.

## Expected Time

Estimated outside-class time: 5-7 hours.

Workload guidance:

- Part A (portfolio tabular representation and model comparison): ~4-5 hours.
- Part B (portfolio checkpoint and model-choice note): ~1-2 hours.

## What You Will Do

Apply MLPs, categorical embeddings, or another appropriate neural tabular approach to your portfolio dataset or to a portfolio-derived tabular proxy. Compare the neural approach against one or more simpler baselines using the same target, split or evaluation procedure, and task-relevant metrics.

If your portfolio project is already tabular, use the portfolio dataset directly. If your portfolio project uses images, text, sequences, audio, graphs, or another non-tabular data type, use an instructor-provided or instructor-approved transformation routine to create a tabular proxy representation. Starter routines are provided in the Week 6 examples zip on the ++[Week 6 Examples and Data page](https://utahtech.instructure.com/courses/1254683/pages/week-6-examples-and-data)++. After downloading and unzipping it, use the `week06_proxy_translators/` folder. The proxy is not your final representation; it is a Week 6 diagnostic tool for deciding what tabular methods can and cannot support.

If your portfolio data is not accessible, not labeled, not legally usable, or otherwise blocked in a way that prevents even a proxy representation, use the shared tabular case study or instructor-approved sample materials only as a fallback. In that case, clearly label Part A as fallback case-study work, explain the blocker, and state what would be needed to create a portfolio-derived tabular proxy.

All students also complete a short portfolio checkpoint so portfolio progress stays connected to the Assignment 4 charter, dataset audit, and staged model-improvement plan.

## Reuse From Assignments 4 and 5

Before starting new work, review your Assignment 4 portfolio charter and Assignment 5 evaluation report.

Use those materials to carry forward:

- The intended stakeholder or client decision.
- The prediction target and candidate inputs.
- The train/validation/test split or other evaluation strategy.
- The metrics and success criteria that fit the task.
- The leakage, prediction-time availability, missingness, imbalance, representativeness, proxy-variable, fairness, or responsible-use risks that need evidence.
- The current baseline or initial model status and next staged model-improvement plan.

If you change the split strategy, metric, baseline, neural model candidate, or scope from Assignment 4 or 5, briefly explain why the change is necessary. Do not redefine the core portfolio project unless the Assignment 4 scope has become infeasible and you have instructor approval.

## Required Work

### Part A: Portfolio tabular representation and model comparison (core work, ~4-5 hours)

Using your portfolio dataset directly, a portfolio-derived tabular proxy, or approved fallback case-study materials:

- State the portfolio problem, client or stakeholder scenario, dataset, prediction target, candidate inputs, and whether Part A uses native tabular portfolio data, a portfolio-derived proxy, or approved fallback case-study materials.
- If using a proxy representation, describe the transformation routine used and what information it preserves or loses.
- If using fallback case-study materials, explain the portfolio-data blocker and what would be needed to create a portfolio-derived proxy.
- Prepare numeric, categorical, missing, rare, or high-cardinality features in a way that avoids leakage and keeps prediction-time availability clear.
- Train at least one simple baseline model, such as logistic regression, linear regression, a tree-based model, or another appropriate classical model.
- Train at least one neural model appropriate for tabular data, such as an MLP or an embedding-based model.
- Continue using the generalization practices from Assignment 5 for neural tabular models. At minimum, use training-split normalization and validation-aware early stopping or checkpoint selection for any MLP. When appropriate, test one regularization option such as dropout, weight decay, or model simplification.
- Use categorical embeddings if the dataset and task make them appropriate; if embeddings are not appropriate, briefly explain why.
- Compare the models using the same target, split or evaluation procedure, and task-relevant metrics. Use validation evidence for model selection and reserve test evidence for final reporting.
- Include a compact results table showing each model, key settings, metrics, and practical notes.
- Discuss practical constraints such as interpretability, cost, training and inference complexity, maintainability, data size, and ease of monitoring.
- Identify at least one responsible-use concern relevant to tabular business prediction, such as sensitive features, proxy variables, fairness across groups, automation bias, or human-review needs.
- Explain whether a tabular approach is justified for the portfolio problem, and whether the neural tabular model is justified over the simpler baseline.

Scope boundary: use one focused comparison rather than a broad model tournament. You may run a small number of diagnostic variants if they clarify the comparison, but do not present the work as an exhaustive hyperparameter search or architecture search.

Depth expectation: this section should make the portfolio model-choice argument inspectable. A reader should be able to see whether the tabular representation was reasonable, whether the neural model was compared fairly, what evidence supports the recommendation, and which practical or responsible-use concerns affect the decision.

### Part B: Portfolio checkpoint and model-choice note (portfolio work, ~1-2 hours)

Summarize what Week 6 implies for your portfolio project, especially whether tabular modeling should remain part of your staged model plan.

Your checkpoint should include:

- Current data readiness.
- Current baseline or initial model status.
- One concrete next planned experiment.
- The expected staged improvement before the final package.
- How the Week 6 evidence affects your final model-choice argument.
- Whether anything in the Assignment 4 dataset audit or charter should be updated, emphasized, or treated as still untested.
- Whether tabular methods, embeddings, or simpler baselines are directly relevant, indirectly relevant, or not relevant to your portfolio project.

Depth expectation: keep this section concise but specific. The purpose is to keep the portfolio thread moving and prepare for Week 7, where you will continue portfolio modeling progress and decide whether transfer learning or image-based methods are relevant.

## Deliverable

Submit one `.zip` file containing code or case-study artifacts and one writeup document with two labeled parts: Part A and Part B.

Required structure:

- Code, notebook, proxy-transformation, or case-study artifacts: scripts, notebooks, configuration notes, command history, saved metrics, plots, or provided sample-result references sufficient to understand what was compared.
- Run evidence: include the command used when you ran the models yourself, the data split description, random seed or nondeterminism note when relevant, saved output, metric file, or other evidence showing that the comparison ran.
- Part A writeup (portfolio tabular representation and model comparison): include portfolio problem summary, stakeholder scenario, native-tabular or proxy-representation notes, preprocessing notes, baseline model, neural tabular model, split or evaluation procedure, metrics, results table, practical constraints, responsible-use concern, and model-choice recommendation.
- Part B writeup (portfolio checkpoint): include current portfolio data readiness, baseline or model status, next experiment, expected staged improvement, model-choice implications, and any needed update to Assignment 4 dataset-audit or charter assumptions.

What is not required in this assignment:

- No exhaustive hyperparameter search.
- No requirement that the neural model outperform the baseline.
- No requirement to use embeddings when the data does not justify them.
- No requirement to test every regularization method; one focused generalization control is enough when training an MLP.
- No requirement to build a CNN, transformer, sequence model, or other modality-specific deep model outside class.
- No requirement that a proxy representation become the final portfolio representation.
- No requirement to make the current portfolio model final.
- No separate outside-class image, sequence, or transformer build.
- No project-scope change unless your Assignment 4 charter has become infeasible and you have instructor approval.

## Portfolio Connection

Your final presentation must justify model choice. This assignment helps you practice saying either "the deep model is worth it" or "the simpler model is better" based on evidence rather than novelty.

For tabular portfolio projects, this assignment can directly strengthen the staged model candidate. For non-tabular portfolio projects, the proxy representation creates a bounded comparison point: it can show that simple structured features are surprisingly useful, that they are inadequate, or that they should be combined later with modality-specific methods introduced in Weeks 7-10.

Assignment 7 will continue this thread by asking for updated portfolio modeling progress and a preliminary decision about whether transfer learning or image-based methods fit your own project.

## Success Criteria

A strong submission compares models fairly, uses appropriate metrics, and gives a practical recommendation. It does not assume that deep learning is automatically better.

A strong submission also keeps the portfolio project moving. It names the current model state, identifies the next experiment, connects Week 6 evidence to the final model-choice argument, and updates or flags any Assignment 4 audit assumptions that the current evidence affects.