package com.example.ia;

import org.junit.jupiter.api.*;
import org.tribuo.*;
import org.tribuo.classification.*;
import org.tribuo.classification.evaluation.*;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
import org.tribuo.data.columnar.FieldProcessor;
import org.tribuo.data.columnar.RowProcessor;
import org.tribuo.data.columnar.processors.field.DoubleFieldProcessor;
import org.tribuo.data.columnar.processors.field.IdentityProcessor;
import org.tribuo.data.columnar.processors.response.FieldResponseProcessor;
import org.tribuo.data.csv.CSVDataSource;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.impl.ArrayExample;

import java.io.*;
import java.nio.file.*;
import java.util.*;

import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.*;

@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class LogistiqueLMIATests {

    private static final Logger logger = Logger.getLogger(LogistiqueLMIATests.class.getName());

    private static final String fileName = "livraison_retards_dataset.csv";
    private static final String newFileName = "livraison_retards_dataset_converted.csv";
    private static final String modelFile = "livraison_regressor.ser";
    private static final Path input = Paths.get("src", "main", "resources", fileName);
    private static final Path output = Paths.get("src", "main", "resources", newFileName);
    private static final Path MODEL_PATH = Paths.get("src", "main", "resources", modelFile);

    private static LabelFactory LabelFactory;
    private static LinkedHashMap<String, FieldProcessor> fieldProcessors;
    private static RowProcessor<Label> rowProcessor;
    private static CSVDataSource<Label> dataSource;
    private static MutableDataset<Label> train;
    private static MutableDataset<Label> test;
    private static Model<Label> model;

    @BeforeAll
    public static void setUp() {
        LabelFactory = new LabelFactory(); // Définir le label factory
        fieldProcessors = new LinkedHashMap<>(); // Définir les extracteurs de colonnes
        configFile(); // encodage des données
    }

    @AfterAll
    public static void tearDown() throws IOException { // Nettoyage des ressources
        Files.deleteIfExists(output);
        Files.deleteIfExists(MODEL_PATH);
    }

    private static void configFile() {
        // nouveau champ calculé prétraité
        fieldProcessors.put("heure_decimal", new DoubleFieldProcessor("heure_decimal"));

        // distance_km est déjà numérique
        fieldProcessors.put("distance_km", new DoubleFieldProcessor("distance_km"));

        // pluie, jour_semaine, vehicule_type = colonnes catégorielles
        fieldProcessors.put("pluie", new IdentityProcessor("pluie"));
        fieldProcessors.put("jour_semaine", new IdentityProcessor("jour_semaine"));
        fieldProcessors.put("vehicule_type", new IdentityProcessor("vehicule_type"));

        // Le processeur de la colonne de sortie (retard)
        FieldResponseProcessor<Label> responseProcessor =
                new FieldResponseProcessor<>("retard", "non", LabelFactory);

        // Création du RowProcessor avec les éléments définis
        rowProcessor = new RowProcessor<>(responseProcessor, fieldProcessors);
    }

    @Test
    @Order(1)
    @DisplayName("Prétraitement : convertir heure_depart en heure_decimal")
    void prepareDatasets() {
        HeureDepartPreprocessor.convertPreprocessor(input, output);
        assertTrue(output.toFile().exists(), "Le fichier converti doit exister");
        assertTrue(output.toFile().length() > 0, "Le fichier converti ne doit pas être vide");
    }

    @Test
    @Order(2)
    @DisplayName("Chargement du dataset CSV")
    void loadDatasets() {
        dataSource = new CSVDataSource<>(
                Paths.get("src", "main", "resources", newFileName),
                rowProcessor,
                true
        );
        assertNotNull(dataSource, "La source de données ne doit pas être null");
        assertFalse(dataSource.toString().isEmpty(), "La source de données doit contenir des données");
    }

    @Test
    @Order(3)
    @DisplayName("Division Train/Test 80/20")
    void splitTrainTest() { // Split train/test => Utiliser 80% des données pour l'entraînement, 20% pour les tests
        var splitter = new TrainTestSplitter<>(dataSource, 0.8, 42L);
        train = new MutableDataset<>(splitter.getTrain());
        test = new MutableDataset<>(splitter.getTest());
    }

    @Test
    @Order(4)
    @DisplayName("Entraînement du modèle (Logistic Regression)")
    void training() { // Entraînement du modèle
        var trainer = new LogisticRegressionTrainer();
        model = trainer.train(train);
    }

    @Test
    @Order(5)
    @DisplayName("Évaluation : accuracy, matrice de confusion, f1-score")
    void evaluator() { // Évaluation => Calculer l'accuracy, matrice de confusion, f1-score
        var evaluator = new LabelEvaluator();
        LabelEvaluation evaluation = evaluator.evaluate(model, test);
        logger.info("Résultats :");
        logger.info(evaluation.toString());
    }

    @Test
    @Order(6)
    @DisplayName("Sauvegarde du modèle")
    void saveModel() throws Exception { // Sauvegarde du modèle
        try (ObjectOutputStream objectOutputStream = new ObjectOutputStream(new FileOutputStream(MODEL_PATH.toFile()))) {
            objectOutputStream.writeObject(model);
        }
    }

    @Test
    @Order(7)
    @DisplayName("Prédiction sur un nouvel échantillon")
    @SuppressWarnings("unchecked")
    void predictor() throws Exception {
        try (ObjectInputStream objectInputStream = new ObjectInputStream(new FileInputStream(MODEL_PATH.toFile()))) {
            Model<Label> loadedModel = (Model<Label>) objectInputStream.readObject();

            // logger.info("Features attendues par le modèle :");
            // loadedModel.getFeatureIDMap().forEach(name -> logger.info(" - " + name));

            Example<Label> example = new ArrayExample<>(new Label("non"));
            // l'exemple doit avoir les mêmes nom de features que ceux générés par le modèle (FieldProcessor)
            example.add(new Feature("distance_km@value", 120.0));
            example.add(new Feature("heure_decimal@value", 8.0));
            example.add(new Feature("pluie@non", 0.0));
            example.add(new Feature("jour_semaine@mercredi", 2.0));
            example.add(new Feature("vehicule_type@camionnette", 1.0));

            // === Prédiction sur l'exemple ===
            var prediction = loadedModel.predict(example);
            logger.info("Prédiction : " + prediction.getOutput());
        }
    }
}
