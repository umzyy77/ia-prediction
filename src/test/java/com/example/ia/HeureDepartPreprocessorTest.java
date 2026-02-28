package com.example.ia;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("HeureDepartPreprocessor - Prétraitement CSV")
class HeureDepartPreprocessorTest {

    @TempDir
    Path tempDir;

    @Test
    @DisplayName("Conversion heure_depart en heure_decimal")
    void shouldConvertHeureDepart() throws IOException {
        Path input = tempDir.resolve("input.csv");
        Path output = tempDir.resolve("output.csv");
        Files.write(input, List.of(
                "heure_depart,distance_km,pluie",
                "08:30,15.0,non",
                "10:00,80.0,oui"
        ));

        HeureDepartPreprocessor.convertPreprocessor(input, output);

        List<String> lines = Files.readAllLines(output);
        assertEquals("heure_decimal,distance_km,pluie", lines.get(0));
        assertTrue(lines.get(1).startsWith("8.5,"));
        assertTrue(lines.get(2).startsWith("10.0,"));
    }

    @Test
    @DisplayName("Les lignes vides sont ignorées")
    void shouldSkipBlankLines() throws IOException {
        Path input = tempDir.resolve("input.csv");
        Path output = tempDir.resolve("output.csv");
        Files.write(input, List.of(
                "heure_depart,distance_km",
                "06:00,20.0",
                "",
                "09:30,50.0"
        ));

        HeureDepartPreprocessor.convertPreprocessor(input, output);

        List<String> lines = Files.readAllLines(output);
        assertEquals(3, lines.size());
    }

    @Test
    @DisplayName("Fichier inexistant lève une RuntimeException")
    void shouldThrowOnInvalidFile() {
        Path input = tempDir.resolve("inexistant.csv");
        Path output = tempDir.resolve("output.csv");

        assertThrows(RuntimeException.class, () ->
                HeureDepartPreprocessor.convertPreprocessor(input, output));
    }
}
