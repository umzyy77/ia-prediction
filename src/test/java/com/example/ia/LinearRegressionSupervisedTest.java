package com.example.ia;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

@DisplayName("LinearRegressionSupervised - Régression linéaire supervisée")
class LinearRegressionSupervisedTest {

    @Test
    @DisplayName("Exécution du main sans erreur")
    void shouldRunMainWithoutError() {
        LinearRegressionSupervised.main(new String[]{});
    }
}
