package com.example.ia;

import org.junit.jupiter.api.*;
import org.tribuo.*;
import org.tribuo.classification.*;
import org.tribuo.classification.evaluation.*;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
import org.tribuo.data.columnar.FieldProcessor;
import org.tribuo.data.columnar.RowProcessor;
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
@DisplayName("TP Pour aller plus loin - Prédiction de la pluie")
public class PluiePredictionTests {

    private static final String newFileName = "livraison_retards_dataset_pluie_converted.csv";
    private static final String modelFileName = "pluie_model.ser";
    private static final Path input = Paths.get("src", "main", "resources", "livraison_retards_dataset.csv");
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
            output.toFile().delete();
        }
        if (MODEL_PATH.toFile().exists()) {
            MODEL_PATH.toFile().delete();
        }
    }

    private static void configFile() {
        // Features : jour_semaine et retard (colonnes catégorielles)
        fieldProcessors.put("jour_semaine", new IdentityProcessor("jour_semaine"));
        fieldProcessors.put("retard", new IdentityProcessor("retard"));

        // La sortie à prédire est maintenant "pluie"
        FieldResponseProcessor<Label> responseProcessor =
                new FieldResponseProcessor<>("pluie", "non", labelFactory);

        rowProcessor = new RowProcessor<>(responseProcessor, fieldProcessors);
    }

    @Test
    @Order(1)
    @DisplayName("Prétraitement des données pour prédiction pluie")
    void prepareDatasets() {
        HeureDepartPreprocessor.convertPreprocessor(input, output);
        assertTrue(output.toFile().exists(), "Le fichier converti doit exister");
    }

    @Test
    @Order(2)
    @DisplayName("Chargement du dataset")
    void loadDatasets() throws IOException {
        dataSource = new CSVDataSource<>(
                Paths.get("src", "main", "resources", newFileName),
                rowProcessor,
                true
        );
        assertNotNull(dataSource);
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
    @DisplayName("Entraînement du modèle de prédiction pluie")
    void training() {
        var trainer = new LogisticRegressionTrainer();
        model = trainer.train(train);
        assertNotNull(model);
    }

    @Test
    @Order(5)
    @DisplayName("Évaluation : accuracy, matrice de confusion, f1-score")
    void evaluator() {
        var evaluator = new LabelEvaluator();
        LabelEvaluation evaluation = evaluator.evaluate(model, test);

        System.out.println("=== Résultats Prédiction Pluie ===");
        System.out.println(evaluation.toString());
        System.out.println("Accuracy : " + evaluation.accuracy());
        System.out.println("Matrice de confusion : " + evaluation.getConfusionMatrix().toString());

        assertTrue(evaluation.accuracy() >= 0, "L'accuracy doit être positive");
    }

    @Test
    @Order(6)
    @DisplayName("Arbre de probabilités")
    void probabilityTree() {
        System.out.println("=== Arbre de probabilités ===");
        for (int i = 0; i < test.size(); i++) {
            var example = test.getExample(i);
            var prediction = model.predict(example);
            System.out.printf("Exemple %d -> Prédiction: %s | Scores: %s%n",
                    i, prediction.getOutput(), prediction.getOutputScores());
        }
    }

    @Test
    @Order(7)
    @DisplayName("Sauvegarde du modèle pluie")
    void saveModel() throws Exception {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(MODEL_PATH.toFile()))) {
            oos.writeObject(model);
        }
        assertTrue(MODEL_PATH.toFile().exists());
    }

    @Test
    @Order(8)
    @DisplayName("Prédiction : vendredi + retard = oui => pluie ?")
    void predictor() throws Exception {
        Model<Label> loadedModel;
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(MODEL_PATH.toFile()))) {
            loadedModel = (Model<Label>) ois.readObject();
        }

        Example<Label> example = new ArrayExample<>(new Label("non"));
        example.add(new Feature("jour_semaine@vendredi", 1.0));
        example.add(new Feature("retard@oui", 1.0));

        var prediction = loadedModel.predict(example);
        System.out.println("=== Prédiction Pluie ===");
        System.out.println("Entrée : jour_semaine=vendredi, retard=oui");
        System.out.println("Prédiction : " + prediction.getOutput());
        System.out.println("Scores : " + prediction.getOutputScores());
        assertNotNull(prediction);
    }
}
