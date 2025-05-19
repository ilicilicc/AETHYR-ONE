package evolver

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io/ioutil"
    "net/http"
    "os"
)

type APIRequest struct {
    Prompt string `json:"prompt"`
}

type APIResponse struct {
    Content string `json:"content"`
}

func loadAPIKeys() (string, string) {
    // Load from env or fallback
    gemini := os.Getenv("GEMINI_API_KEY")
    xai := os.Getenv("XAI_API_KEY")
    return gemini, xai
}

func QueryGemini(prompt string) string {
    geminiKey, _ := loadAPIKeys()
    body := map[string]interface{}{
        "contents": []map[string]string{
            {"role": "user", "parts": prompt},
        },
    }
    jsonData, _ := json.Marshal(body)

    req, _ := http.NewRequest("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key="+geminiKey, bytes.NewBuffer(jsonData))
    req.Header.Set("Content-Type", "application/json")
    client := &http.Client{}
    resp, err := client.Do(req)
    if err != nil {
        return "Error contacting Gemini"
    }
    defer resp.Body.Close()
    bodyBytes, _ := ioutil.ReadAll(resp.Body)
    return string(bodyBytes)
}

func QueryXAI(prompt string) string {
    _, xaiKey := loadAPIKeys()
    body := APIRequest{Prompt: prompt}
    jsonData, _ := json.Marshal(body)

    req, _ := http.NewRequest("POST", "https://xai-api-url.com/generate?key="+xaiKey, bytes.NewBuffer(jsonData))
    req.Header.Set("Content-Type", "application/json")
    client := &http.Client{}
    resp, err := client.Do(req)
    if err != nil {
        return "Error contacting XAI"
    }
    defer resp.Body.Close()
    bodyBytes, _ := ioutil.ReadAll(resp.Body)
    return string(bodyBytes)
}
