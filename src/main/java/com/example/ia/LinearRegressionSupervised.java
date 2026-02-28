package com.example.ia;

// Objectif pédagogique => Implémenter une régression linéaire supervisée
public class LinearRegressionSupervised {
    public static void main(String[] args) {
        // On apprend à prédire une sortie Y à partir d'une entrée X avec une fct du type Y = a * X + b.
        double[] xData = {6, 8, 10, 12}; // Données d'entraînement => X: heure
        double[] yData = {0, 0, 1, 1};   // Données d'entraînement => Y: retard oui=1 / non=0
        double a = 0.0; // pente
        double b = 0.0; // intercept
        double learningRate = 0.01;

        for (int epoch = 0; epoch < 1000; epoch++) { // Entraînement (gradient descent)
            double totalErrorA = 0;
            double totalErrorB = 0;

            for (int i = 0; i < xData.length; i++) {
                double x = xData[i];
                double y = yData[i];
                double prediction = a * x + b;
                double error = prediction - y;
                totalErrorA += error * x;
                totalErrorB += error;
            }

            a -= learningRate * totalErrorA / xData.length; // Mise à jour de la pente
            b -= learningRate * totalErrorB / xData.length; // Mise à jour de la constante intercept
        }
        System.out.printf("Modèle entraîné : y = %.3f * x + %.3f%n", a, b); // Affichage du modèle appris

        double testHour = 11; // Test : prédire le retard pour 11h
        System.out.printf("Probabilité de retard à %.0fh : %.2f%n", testHour, a * testHour + b);
    }
}
