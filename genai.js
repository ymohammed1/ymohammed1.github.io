/*
====================================================
    Author: Prashant Desale
    Date: 03/05/2025
    File: genai.js
    Description: 
    This JavaScript file manages GenAI task execution using the Hugging Face Transformers library. It includes:
      - Configuration for multiple AI tasks (Sentiment Analysis, NER, QA, Summarization, Translation, and Text Generation).
      - UI interaction handling (task selection, input processing).
      - Model loading and execution logic.
      - Result rendering with error handling.
====================================================
*/

// Import the pipeline function from the Hugging Face Transformers library
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

// Ensure that local models are not used (force cloud-based execution)
env.allowLocalModels = false;

/* 
    GenAI Task Configuration:
    - Defines various GenAI tasks, their associated models, descriptions, example inputs, and sample code snippets.
*/
const taskConfigs = {
    'sentiment': {
        name: 'sentiment-analysis',
        model: 'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
        description: 'Analyzes the sentiment (positive/negative) of the provided text.',
        sampleInput: 'I really enjoyed this movie, it was fantastic!',
        code: 'const result = await pipeline("sentiment-analysis")("INPUT_TEXT");'
    },
    'ner': {
        name: 'token-classification',
        model: 'Xenova/bert-base-NER',
        description: 'Identifies named entities (people, locations, organizations, etc.) in the text.',
        sampleInput: 'WEX CEO Melissa Smith visited Dallas last week.',
        code: 'const result = await pipeline("token-classification")("INPUT_TEXT");'
    },
    'qa': {
        name: 'question-answering',
        model: 'Xenova/distilbert-base-uncased-distilled-squad',
        description: 'Finds the answer to a question within the provided context.',
        sampleInput: 'The Eiffel Tower was completed in 1889.\nQUESTION: When was the Eiffel Tower completed?',
        code: 'const result = await pipeline("question-answering")({ question: "QUESTION", context: "CONTEXT" });'
    },
    'summary': {
        name: 'summarization',
        model: 'Xenova/distilbart-cnn-6-6',
        description: 'Creates a concise summary of a longer text.',
        sampleInput: 'The United States of America, founded in 1776, is a federal republic composed of 50 states, a capital district (Washington, D.C.), and several territories. It operates under a system of democracy with a constitution that establishes a separation of powers among the executive, legislative, and judicial branches of government. The country has played a significant role in global politics, economics, and culture, emerging as one of the worlds most influential nations.Known for its diverse population, the U.S.is a melting pot of cultures, traditions, and languages, shaped by immigration and historical developments.It boasts a strong economy driven by industries such as technology, finance, healthcare, and entertainment.Additionally, the U.S.is home to some of the most prestigious educational institutions and research centers, contributing to innovation and scientific advancements.With vast geographical landscapes ranging from bustling urban centers to breathtaking natural wonders, the nation remains a global leader in various fields, including space exploration, military power, and human rights advocacy.',
        code: 'const result = await pipeline("summarization")("INPUT_TEXT", { max_length: 100 });'
    },
    'translate': {
        name: 'translation',
        model: 'Xenova/m2m100_418M',
        description: 'Translates text from English to Hindi.',
        sampleInput: 'Hello, how are you?',
        code: 'const result = await pipeline("translation")("INPUT_TEXT", { src_lang: "en", tgt_lang: "hi" });'
    },
    'generate': {
        name: 'text-generation',
        model: 'Xenova/distilgpt2',
        description: 'Generates text continuation based on input.',
        sampleInput: 'My favorite food is',
        code: 'const result = await pipeline("text-generation")("INPUT_TEXT", { max_length: 50 });'
    }
};

// Object to store loaded models for reuse
let taskModels = {};

// Variables for UI elements
let userInput, runButton, codePreview, taskDescription, taskStatus, taskResult;

/* 
    Function: initializeElements()
    - Retrieves DOM elements required for user interaction.
    - Ensures all required elements exist before proceeding.
*/
function initializeElements() {
    userInput = document.getElementById('userInput');
    runButton = document.getElementById('runTask');
    codePreview = document.getElementById('codePreview');
    taskDescription = document.getElementById('taskDescription');
    taskStatus = document.getElementById('taskStatus');
    taskResult = document.getElementById('taskResult');

    if (!userInput || !runButton || !codePreview || !taskDescription || !taskStatus || !taskResult) {
        console.error('Some elements were not found in the DOM');
        return false;
    }
    return true;
}

/* 
    Function: parseQAInput(inputText)
    - Parses the input text to separate context and question for the Question & Answer task.
    - Ensures correct formatting before sending to the AI model.
*/
function parseQAInput(inputText) {
    if (typeof inputText !== 'string') {
        throw new Error('Input must be a text string');
    }

    const questionIndex = inputText.indexOf('\nQUESTION:');
    if (questionIndex === -1) {
        throw new Error('Please add "QUESTION:" on a new line before your question.');
    }

    const context = inputText.substring(0, questionIndex).trim();
    const question = inputText.substring(questionIndex + 10).trim();

    if (!context || !question) {
        throw new Error('Both context and question are required.');
    }

    return { context, question };
}

/* 
    Function: initialize()
    - Initializes the application, sets default task, and adds event listeners.
*/
function initialize() {
    if (!initializeElements()) return;

    // Set default task to sentiment analysis
    updateTaskInfo('sentiment');

    // Event listener for task selection change
    document.querySelectorAll('input[name="task"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            updateTaskInfo(e.target.value);
        });
    });

    // Event listener for running a task
    runButton.addEventListener('click', async () => {
        const selectedTask = document.querySelector('input[name="task"]:checked').value;
        const config = taskConfigs[selectedTask];
        let text = userInput.value || config.sampleInput;

        text = String(text);

        taskStatus.textContent = `Status: Loading ${config.name} model...`;
        taskStatus.classList.add('loading');
        taskResult.innerHTML = '';

        try {
            if (!taskModels[selectedTask]) {
                taskModels[selectedTask] = await pipeline(config.name, config.model);
            }

            let result;
            switch (selectedTask) {
                case 'sentiment':
                    result = await taskModels[selectedTask](text);
                    taskResult.innerHTML = `<strong>Sentiment:</strong> ${result[0].label}<br>
                                            <strong>Confidence:</strong> ${(result[0].score * 100).toFixed(2)}%`;
                    break;

                case 'ner':
                    result = await taskModels[selectedTask](text);
                    displayNERResults(result);
                    break;

                case 'qa':
                    const { context, question } = parseQAInput(text);
                    result = await taskModels[selectedTask](question, context);
                    taskResult.innerHTML = `<strong>Question:</strong> ${question}<br>
                                            <strong>Answer:</strong> ${result.answer}<br>
                                            <strong>Confidence:</strong> ${(result.score * 100).toFixed(2)}%`;
                    break;

                case 'summary':
                    result = await taskModels[selectedTask](text, { max_length: 100 });
                    taskResult.innerHTML = `<strong>Summary:</strong> ${result[0].summary_text}`;
                    break;

                case 'translate':
                    result = await taskModels[selectedTask](text, { src_lang: 'en', tgt_lang: 'hi' });
                    taskResult.innerHTML = `<strong>Translation:</strong> ${result[0].translation_text}`;
                    break;

                case 'generate':
                    result = await taskModels[selectedTask](text, { max_length: 50 });
                    taskResult.innerHTML = `<strong>Generated Text:</strong> ${result[0].generated_text}`;
                    break;
            }

            taskStatus.textContent = 'Status: Complete';
            taskStatus.classList.remove('loading');
        } catch (error) {
            console.error('Task Error:', error);
            taskStatus.textContent = 'Status: Error';
            taskResult.innerHTML = `<div style="color: red;">Error: ${error.message}</div>`;
        }
    });
}

/* 
    Function: updateTaskInfo(taskKey)
    - Updates the displayed task information when a new task is selected.
*/
function updateTaskInfo(taskKey) {
    const config = taskConfigs[taskKey];
    if (codePreview && taskDescription && userInput) {
        codePreview.textContent = config.code;
        taskDescription.textContent = config.description;
        userInput.value = config.sampleInput;

        clearResults();
    }
}

/* 
    Function: clearResults()
    - Resets the task status and clears previous results.
*/
function clearResults() {
    if (taskStatus && taskResult) {
        taskStatus.textContent = 'Status: Ready';
        taskStatus.classList.remove('loading');
        taskResult.innerHTML = '';
    }
}

/* 
    Function: displayNERResults()
    - Function to display NER Results.
*/
function displayNERResults(results) {
    if (!taskResult) return;

    const table = `
        <table>
            <thead>
                <tr>
                    <th>Entity</th>
                    <th>Score</th>
                    <th>Index</th>
                    <th>Word</th>
                </tr>
            </thead>
            <tbody>
                ${results.map(item => `
                    <tr>
                        <td>${item.entity}</td>
                        <td>${(item.score * 100).toFixed(2)}%</td>
                        <td>${item.index}</td>
                        <td>${item.word}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
    taskResult.innerHTML = table;
}
// Initialize script when the DOM is loaded
document.addEventListener('DOMContentLoaded', initialize);
