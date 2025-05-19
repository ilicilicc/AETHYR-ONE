package evolver

func ScoreResult(output string) int {
    // Simple scoring based on length of response
    return len(output)
}
