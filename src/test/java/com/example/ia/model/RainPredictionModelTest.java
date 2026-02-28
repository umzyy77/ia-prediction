package com.example.ia.model;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.tribuo.classification.Label;
import org.tribuo.data.columnar.RowProcessor;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("RainPredictionModel - Configuration du modèle")
class RainPredictionModelTest {

    @Test
    @DisplayName("Le RowProcessor doit être correctement configuré")
    void rowProcessorShouldBeConfigured() {
        RainPredictionModel model = new RainPredictionModel();
        RowProcessor<Label> rowProcessor = model.getRowProcessor();

        assertNotNull(rowProcessor, "Le RowProcessor ne doit pas être null");
    }

    @Test
    @DisplayName("Le modèle doit utiliser pluie comme variable de sortie")
    void shouldUsePluieAsOutput() {
        RainPredictionModel model = new RainPredictionModel();
        RowProcessor<Label> rowProcessor = model.getRowProcessor();

        String description = rowProcessor.getResponseProcessor().toString();
        assertTrue(description.contains("pluie"), "La variable de sortie doit être 'pluie'");
    }
}
