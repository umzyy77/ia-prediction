package com.example.ia.analysis;

import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.classification.evaluation.LabelEvaluation;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

public class RainCategoricalAnalysis {

    private static final Logger logger = Logger.getLogger(RainCategoricalAnalysis.class.getName());

    public record ProbabilityEntry(int index, Label prediction, Map<String, Label> scores) {}

    public List<ProbabilityEntry> buildProbabilityTree(Model<Label> model, MutableDataset<Label> testData) {
        List<ProbabilityEntry> entries = new ArrayList<>();
        for (int i = 0; i < testData.size(); i++) {
            var example = testData.getExample(i);
            var prediction = model.predict(example);
            entries.add(new ProbabilityEntry(i, prediction.getOutput(), prediction.getOutputScores()));
        }
        return entries;
    }

    public void printEvaluationReport(LabelEvaluation evaluation) {
        logger.info("=== Résultats Prédiction Pluie ===");
        logger.info(evaluation.toString());
        logger.info("Accuracy : " + evaluation.accuracy());
        logger.info("Matrice de confusion :\n" + evaluation.getConfusionMatrix().toString());
    }

    public void printProbabilityTree(List<ProbabilityEntry> entries) {
        logger.info("=== Arbre de probabilités ===");
        for (var entry : entries) {
            logger.info(String.format("Exemple %d -> Prédiction: %s | Scores: %s",
                    entry.index(), entry.prediction(), entry.scores()));
        }
    }

    public void printPrediction(Prediction<Label> prediction, String jourSemaine, String retard) {
        logger.info("=== Prédiction Pluie ===");
        logger.info(String.format("Entrée : jour_semaine=%s, retard=%s", jourSemaine, retard));
        logger.info("Prédiction : " + prediction.getOutput());
        logger.info("Scores : " + prediction.getOutputScores());
    }
}
