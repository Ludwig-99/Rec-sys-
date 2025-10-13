let reviews = [];
let currentReview = null;

document.addEventListener('DOMContentLoaded', async () => {
    const savedToken = localStorage.getItem('hfToken');
    if (savedToken) {
        document.getElementById('token').value = savedToken;
    }
    
    document.getElementById('token').addEventListener('input', function() {
        localStorage.setItem('hfToken', this.value);
    });
    
    await loadReviews();
    
    document.getElementById('randomReview').addEventListener('click', selectRandomReview);
    document.getElementById('analyzeSentiment').addEventListener('click', analyzeSentiment);
    document.getElementById('countNouns').addEventListener('click', countNouns);
});

async function loadReviews() {
    try {
        const response = await fetch('reviews_test.tsv');
        const tsvData = await response.text();
        
        const parsed = Papa.parse(tsvData, {
            header: true,
            delimiter: '\t',
            skipEmptyLines: true
        });
        
        reviews = parsed.data.filter(review => review.text && review.text.trim() !== '');
    } catch (error) {
        showError('Failed to load reviews data: ' + error.message);
    }
}

function selectRandomReview() {
    if (reviews.length === 0) {
        showError('No reviews available');
        return;
    }
    
    const randomIndex = Math.floor(Math.random() * reviews.length);
    currentReview = reviews[randomIndex];
    document.getElementById('reviewText').textContent = currentReview.text;
    
    resetResults();
    hideError();
}

async function analyzeSentiment() {
    if (!currentReview) {
        showError('Please select a review first');
        return;
    }
    
    const token = document.getElementById('token').value.trim();
    
    if (!token) {
        showError('API token required for sentiment analysis');
        return;
    }
    
    await callSentimentApi(currentReview.text, token);
}

function countNouns() {
    if (!currentReview) {
        showError('Please select a review first');
        return;
    }
    
    const doc = nlp(currentReview.text);
    const nouns = doc.nouns().out('array');
    const nounCount = nouns.length;
    
    let level = 'Low';
    let icon = '🔴';
    
    if (nounCount >= 8) {
        level = 'High';
        icon = '🟢';
    } else if (nounCount >= 4) {
        level = 'Medium';
        icon = '🟡';
    }
    
    document.getElementById('nounResult').textContent = icon;
    document.getElementById('nounCount').textContent = `${nounCount} nouns`;
    
    highlightNounsInText(nouns);
}

function highlightNounsInText(nouns) {
    const reviewElement = document.getElementById('reviewText');
    let text = currentReview.text;
    
    nouns.forEach(noun => {
        const regex = new RegExp(`\\b${noun}\\b`, 'gi');
        text = text.replace(regex, `<span class="highlight-noun">${noun}</span>`);
    });
    
    reviewElement.innerHTML = text;
}

async function callSentimentApi(text, token) {
    const spinner = document.getElementById('spinner');
    const errorDiv = document.getElementById('error');
    
    hideError();
    spinner.style.display = 'block';
    disableButtons(true);
    
    try {
        const response = await fetch('https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ inputs: text })
        });
        
        if (response.status === 402) {
            throw new Error('API token required for this model');
        }
        
        if (response.status === 429) {
            throw new Error('Rate limit exceeded. Please try again later.');
        }
        
        if (response.status === 503) {
            throw new Error('Model is loading. Please try again in a few moments.');
        }
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        if (data && data[0]) {
            updateSentimentResult(data[0]);
        } else {
            throw new Error('Unexpected API response format');
        }
        
    } catch (error) {
        showError(error.message);
    } finally {
        spinner.style.display = 'none';
        disableButtons(false);
    }
}

function updateSentimentResult(data) {
    let sentiment = 'Neutral';
    let icon = '❓';
    let confidence = 0;
    
    if (data && data.length > 0) {
        const scores = {};
        data.forEach(item => {
            scores[item.label] = item.score;
        });
        
        const maxLabel = Object.keys(scores).reduce((a, b) => scores[a] > scores[b] ? a : b);
        confidence = (scores[maxLabel] * 100).toFixed(1);
        
        if (maxLabel === 'positive' || maxLabel === 'POSITIVE' || maxLabel === 'LABEL_2') {
            sentiment = 'Positive';
            icon = '👍';
        } else if (maxLabel === 'negative' || maxLabel === 'NEGATIVE' || maxLabel === 'LABEL_0') {
            sentiment = 'Negative';
            icon = '👎';
        } else {
            sentiment = 'Neutral';
            icon = '❓';
        }
    }
    
    document.getElementById('sentimentResult').textContent = icon;
    document.getElementById('sentimentConfidence').textContent = `${confidence}% confidence`;
}

function resetResults() {
    document.getElementById('sentimentResult').textContent = '❓';
    document.getElementById('nounResult').textContent = '❓';
    document.getElementById('sentimentConfidence').textContent = '';
    document.getElementById('nounCount').textContent = '';
    
    if (currentReview) {
        document.getElementById('reviewText').textContent = currentReview.text;
    }
}

function disableButtons(disabled) {
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
        button.disabled = disabled;
    });
}

function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    
    setTimeout(() => {
        hideError();
    }, 5000);
}

function hideError() {
    const errorDiv = document.getElementById('error');
    errorDiv.style.display = 'none';
}
