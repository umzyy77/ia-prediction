package com.example.ia.analysis;

import com.example.ia.HeureDepartPreprocessor;
import com.example.ia.model.RainPredictionModel;
import com.example.ia.service.RainPredictionService;
import org.junit.jupiter.api.*;
import org.tribuo.classification.evaluation.LabelEvaluation;

import java.nio.file.*;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
@DisplayName("RainCategoricalAnalysis - Analyse catégorielle et arbre de probabilités")
class RainCategoricalAnalysisTest {

    private static final Path input = Paths.get("src", "main", "resources", "livraison_retards_dataset.csv");
    private static final Path convertedFile = Paths.get("src", "main", "resources", "livraison_retards_dataset_analysis_converted.csv");

    private static RainPredictionService service;
    private static RainCategoricalAnalysis analysis;

    @BeforeAll
    static void setUp() {
        service = new RainPredictionService();
        analysis = new RainCategoricalAnalysis();

        HeureDepartPreprocessor.convertPreprocessor(input, convertedFile);
        var model = new RainPredictionModel();
        var dataSource = service.loadData(convertedFile, model.getRowProcessor());
        service.splitTrainTest(dataSource, 0.8, 42L);
        service.train();
    }

    @AfterAll
    static void tearDown() {
        if (convertedFile.toFile().exists()) convertedFile.toFile().delete();
    }

    @Test
    @Order(1)
    @DisplayName("Rapport d'évaluation : accuracy, matrice de confusion, f1-score")
    void shouldPrintEvaluationReport() {
        LabelEvaluation evaluation = service.evaluate();
        analysis.printEvaluationReport(evaluation);
        assertTrue(evaluation.accuracy() >= 0);
    }

    @Test
    @Order(2)
    @DisplayName("Arbre de probabilités")
    void shouldBuildAndPrintProbabilityTree() {
        var entries = analysis.buildProbabilityTree(service.getModel(), service.getTest());
        analysis.printProbabilityTree(entries);
        assertFalse(entries.isEmpty(), "L'arbre de probabilités ne doit pas être vide");
    }

    @Test
    @Order(3)
    @DisplayName("Prédiction vendredi + retard=oui avec affichage")
    void shouldPrintPrediction() {
        var prediction = service.predict(service.getModel(), "vendredi", "oui");
        analysis.printPrediction(prediction, "vendredi", "oui");
        assertNotNull(prediction.getOutput());
    }
}
