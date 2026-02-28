package com.example.ia.model;

import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.data.columnar.FieldProcessor;
import org.tribuo.data.columnar.RowProcessor;
import org.tribuo.data.columnar.processors.field.IdentityProcessor;
import org.tribuo.data.columnar.processors.response.FieldResponseProcessor;

import java.util.LinkedHashMap;

public class RainPredictionModel {

    private final RowProcessor<Label> rowProcessor;

    public RainPredictionModel() {
        LabelFactory labelFactory = new LabelFactory();
        LinkedHashMap<String, FieldProcessor> fieldProcessors = new LinkedHashMap<>();

        // Features catégorielles : jour_semaine et retard
        fieldProcessors.put("jour_semaine", new IdentityProcessor("jour_semaine"));
        fieldProcessors.put("retard", new IdentityProcessor("retard"));

        // Sortie à prédire : pluie
        FieldResponseProcessor<Label> responseProcessor =
                new FieldResponseProcessor<>("pluie", "non", labelFactory);

        this.rowProcessor = new RowProcessor<>(responseProcessor, fieldProcessors);
    }

    public RowProcessor<Label> getRowProcessor() {
        return rowProcessor;
    }
}
