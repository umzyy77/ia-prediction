package com.example.ia.service;

import com.example.ia.HeureDepartPreprocessor;
import com.example.ia.model.RainPredictionModel;
import org.junit.jupiter.api.*;
import org.tribuo.Model;
import org.tribuo.classification.Label;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.data.csv.CSVDataSource;

import java.nio.file.*;

import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.*;

@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
@DisplayName("RainPredictionService - Pipeline ML pluie")
class RainPredictionServiceTest {

    private static final Logger logger = Logger.getLogger(RainPredictionServiceTest.class.getName());

    private static final Path input = Paths.get("src", "main", "resources", "livraison_retards_dataset.csv");
    private static final Path convertedFile = Paths.get("src", "main", "resources", "livraison_retards_dataset_pluie_converted.csv");
    private static final Path modelPath = Paths.get("src", "main", "resources", "pluie_model.ser");

    private static RainPredictionService service;
    private static RainPredictionModel rainModel;
    private static CSVDataSource<Label> dataSource;

    @BeforeAll
    static void setUp() {
        service = new RainPredictionService();
        rainModel = new RainPredictionModel();
        HeureDepartPreprocessor.convertPreprocessor(input, convertedFile);
    }

    @AfterAll
    static void tearDown() {
        if (convertedFile.toFile().exists()) convertedFile.toFile().delete();
        if (modelPath.toFile().exists()) modelPath.toFile().delete();
    }

    @Test
    @Order(1)
    @DisplayName("Chargement des données")
    void shouldLoadData() {
        dataSource = service.loadData(convertedFile, rainModel.getRowProcessor());
        assertNotNull(dataSource);
    }

    @Test
    @Order(2)
    @DisplayName("Division Train/Test 80/20")
    void shouldSplitTrainTest() {
        service.splitTrainTest(dataSource, 0.8, 42L);
        assertTrue(service.getTrain().size() > 0);
        assertTrue(service.getTest().size() > 0);
    }

    @Test
    @Order(3)
    @DisplayName("Entraînement du modèle")
    void shouldTrainModel() {
        Model<Label> model = service.train();
        assertNotNull(model);
    }

    @Test
    @Order(4)
    @DisplayName("Évaluation : accuracy, matrice de confusion, f1-score")
    void shouldEvaluateModel() {
        LabelEvaluation evaluation = service.evaluate();
        logger.info(evaluation.toString());
        assertTrue(evaluation.accuracy() >= 0);
    }

    @Test
    @Order(5)
    @DisplayName("Sauvegarde et chargement du modèle")
    void shouldSaveAndLoadModel() throws Exception {
        service.saveModel(modelPath);
        assertTrue(modelPath.toFile().exists());

        Model<Label> loaded = service.loadModel(modelPath);
        assertNotNull(loaded);
    }

    @Test
    @Order(6)
    @DisplayName("Prédiction : vendredi + retard=oui => pluie ?")
    void shouldPredictRain() throws Exception {
        Model<Label> loaded = service.loadModel(modelPath);
        var prediction = service.predict(loaded, "vendredi", "oui");

        assertNotNull(prediction);
        logger.info("Prédiction : " + prediction.getOutput());
        logger.info("Scores : " + prediction.getOutputScores());
    }
}
