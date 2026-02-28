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

import static org.junit.jupiter.api.Assertions.*;

@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class LogistiqueLMIATests {

    private static final String fileName = "livraison_retards_dataset.csv";
    private static final String newFileName = "livraison_retards_dataset_converted.csv";
    private static final String modelFileName = "livraison_regressor.ser";
    private static final Path input = Paths.get("src", "main", "resources", fileName);
    private static final Path output = Paths.get("src", "main", "resources", newFileName);
    private static final Path MODEL_PATH = Paths.get("src", "main", "resources", modelFileName);

    private static LabelFactory labelFactory;
    private static LinkedHashMap<String, FieldProcessor> fieldProcessors;
    private static RowProcessor<Label> rowProcessor;
    private static CSVDataSource<Label> dataSource;
    private static MutableDataset<Label> train;
    private static MutableDataset<Label> test;
    private static Model<Label> model;

    @BeforeAll
    public static void setUp() {
        labelFactory = new LabelFactory();
        fieldProcessors = new LinkedHashMap<>();
        configFile();
    }

    @AfterAll
    public static void tearDown() {
        if (output.toFile().exists()) {
            boolean deleted = output.toFile().delete();
            assertTrue(deleted, "Le fichier converti doit être supprimé après les tests");
        }
        if (MODEL_PATH.toFile().exists()) {
            boolean deleted = MODEL_PATH.toFile().delete();
            assertTrue(deleted, "Le fichier du modèle doit être supprimé après les tests");
        }
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
                new FieldResponseProcessor<>("retard", "non", labelFactory);

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
    void loadDatasets() throws IOException {
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
    void splitTrainTest() {
        var splitter = new TrainTestSplitter<>(dataSource, 0.8, 42L);
        train = new MutableDataset<>(splitter.getTrain());
        test = new MutableDataset<>(splitter.getTest());
        assertTrue(train.size() > 0, "Le dataset d'entraînement ne doit pas être vide");
        assertTrue(test.size() > 0, "Le dataset de test ne doit pas être vide");
    }

    @Test
    @Order(4)
    @DisplayName("Entraînement du modèle (Logistic Regression)")
    void training() {
        var trainer = new LogisticRegressionTrainer();
        model = trainer.train(train);
        assertNotNull(model, "Le modèle entraîné ne doit pas être null");
    }

    @Test
    @Order(5)
    @DisplayName("Évaluation : accuracy, matrice de confusion, f1-score")
    void evaluator() {
        var evaluator = new LabelEvaluator();
        LabelEvaluation evaluation = evaluator.evaluate(model, test);
        System.out.println("Résultats :");
        System.out.println(evaluation.toString());
        assertTrue(evaluation.accuracy() >= 0, "L'accuracy doit être positive");
    }

    @Test
    @Order(6)
    @DisplayName("Sauvegarde du modèle")
    void saveModel() throws Exception {
        try (ObjectOutputStream objectOutputStream = new ObjectOutputStream(new FileOutputStream(MODEL_PATH.toFile()))) {
            objectOutputStream.writeObject(model);
        }
        assertTrue(MODEL_PATH.toFile().exists(), "Le fichier du modèle doit exister");
    }

    @Test
    @Order(7)
    @DisplayName("Prédiction sur un nouvel échantillon")
    void predictor() throws Exception {
        File file = MODEL_PATH.toFile();
        Model<Label> loadedModel;
        try (ObjectInputStream objectInputStream = new ObjectInputStream(new FileInputStream(file))) {
            loadedModel = (Model<Label>) objectInputStream.readObject();
        }

        Example<Label> example = new ArrayExample<>(new Label("non"));
        example.add(new Feature("distance_km@value", 120.0));
        example.add(new Feature("heure_decimal@value", 8.0));
        example.add(new Feature("pluie@non", 0.0));
        example.add(new Feature("jour_semaine@mercredi", 2.0));
        example.add(new Feature("vehicule_type@camionnette", 1.0));

        // === Prédiction sur l'exemple ===
        var prediction = loadedModel.predict(example);
        System.out.println("Prédiction : " + prediction.getOutput());
        assertNotNull(prediction, "La prédiction ne doit pas être null");
    }
}
