package com.example.ia;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class HeureDepartPreprocessor {

    private HeureDepartPreprocessor() {}

    public static void convertPreprocessor(Path input, Path output) {
        try {
            List<String> lines = Files.readAllLines(input);
            List<String> convertedLines = new ArrayList<>();

            // Header : remplacer heure_depart par heure_decimal
            String header = lines.getFirst().replace("heure_depart", "heure_decimal");
            convertedLines.add(header);

            for (int i = 1; i < lines.size(); i++) {
                String line = lines.get(i);
                if (line.isBlank()) continue;

                String[] parts = line.split(",");
                // Convertir heure_depart (HH:mm) en format décimal
                String heureDepart = parts[0];
                double heureDecimal = convertHeureToDecimal(heureDepart);

                // Reconstruire la ligne avec heure_decimal à la place de heure_depart
                parts[0] = String.valueOf(heureDecimal);
                convertedLines.add(String.join(",", parts));
            }

            Files.write(output, convertedLines);
        } catch (IOException e) {
            throw new RuntimeException("Erreur lors du prétraitement du fichier CSV", e);
        }
    }

    private static double convertHeureToDecimal(String heure) {
        String[] parts = heure.split(":");
        int heures = Integer.parseInt(parts[0]);
        int minutes = Integer.parseInt(parts[1]);
        return heures + minutes / 60.0;
    }
}
