package main

import (
    "fmt"
    "github.com/aethyr/evolver/internal/evolver"
)

func main() {
    prompt := "Explain quantum computing in simple terms."
    geminiResponse := evolver.QueryGemini(prompt)
    xaiResponse := evolver.QueryXAI(prompt)

    fmt.Println("Gemini:", geminiResponse)
    fmt.Println("XAI:", xaiResponse)

    scoreGemini := evolver.ScoreResult(geminiResponse)
    scoreXAI := evolver.ScoreResult(xaiResponse)

    fmt.Printf("Gemini Score: %d\n", scoreGemini)
    fmt.Printf("XAI Score: %d\n", scoreXAI)
}
