package com.example.ia.service;

import org.tribuo.*;
import org.tribuo.classification.Label;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.sgd.linear.LogisticRegressionTrainer;
import org.tribuo.data.columnar.RowProcessor;
import org.tribuo.data.csv.CSVDataSource;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.impl.ArrayExample;

import java.io.*;
import java.nio.file.Path;

public class RainPredictionService {

    private MutableDataset<Label> train;
    private MutableDataset<Label> test;
    private Model<Label> model;

    public CSVDataSource<Label> loadData(Path csvPath, RowProcessor<Label> rowProcessor) {
        return new CSVDataSource<>(csvPath, rowProcessor, true);
    }

    public void splitTrainTest(CSVDataSource<Label> dataSource, double trainProportion, long seed) {
        var splitter = new TrainTestSplitter<>(dataSource, trainProportion, seed);
        this.train = new MutableDataset<>(splitter.getTrain());
        this.test = new MutableDataset<>(splitter.getTest());
    }

    public Model<Label> train() {
        var trainer = new LogisticRegressionTrainer();
        this.model = trainer.train(train);
        return model;
    }

    public LabelEvaluation evaluate() {
        var evaluator = new LabelEvaluator();
        return evaluator.evaluate(model, test);
    }

    public void saveModel(Path path) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path.toFile()))) {
            oos.writeObject(model);
        }
    }

    @SuppressWarnings("unchecked")
    public Model<Label> loadModel(Path path) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path.toFile()))) {
            return (Model<Label>) ois.readObject();
        }
    }

    public Prediction<Label> predict(Model<Label> model, String jourSemaine, String retard) {
        Example<Label> example = new ArrayExample<>(new Label("non"));
        example.add(new Feature("jour_semaine@" + jourSemaine, 1.0));
        example.add(new Feature("retard@" + retard, 1.0));
        return model.predict(example);
    }

    public MutableDataset<Label> getTrain() {
        return train;
    }

    public MutableDataset<Label> getTest() {
        return test;
    }

    public Model<Label> getModel() {
        return model;
    }
}
