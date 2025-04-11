// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
  // Get DOM elements
  const checkPageButton = document.getElementById('checkPage');
  const settingsButton = document.getElementById('settingsBtn');
  const settingsPanel = document.getElementById('settingsPanel');
  const saveSettingsButton = document.getElementById('saveSettings');
  const confidenceThreshold = document.getElementById('confidenceThreshold');
  const confidenceValue = document.getElementById('confidenceValue');
  const loader = document.getElementById('loader');
  const results = document.getElementById('results');
  const summary = document.getElementById('summary');
  const factsList = document.getElementById('facts');

  // Load saved settings
  chrome.storage.sync.get({
    confidenceThreshold: 70,
    sources: ['wikipedia', 'news', 'factCheckers']
  }, function(items) {
    confidenceThreshold.value = items.confidenceThreshold;
    confidenceValue.textContent = items.confidenceThreshold + '%';
    
    // Select the saved sources
    const sourcesSelect = document.getElementById('checkingSources');
    for (let i = 0; i < sourcesSelect.options.length; i++) {
      sourcesSelect.options[i].selected = items.sources.includes(sourcesSelect.options[i].value);
    }
  });

  // Update confidence value display when slider changes
  confidenceThreshold.addEventListener('input', function() {
    confidenceValue.textContent = this.value + '%';
  });

  // Toggle settings panel visibility
  settingsButton.addEventListener('click', function() {
    settingsPanel.style.display = settingsPanel.style.display === 'none' ? 'block' : 'none';
  });

  // Save settings
  saveSettingsButton.addEventListener('click', function() {
    const sourcesSelect = document.getElementById('checkingSources');
    const selectedSources = Array.from(sourcesSelect.selectedOptions).map(option => option.value);
    
    chrome.storage.sync.set({
      confidenceThreshold: confidenceThreshold.value,
      sources: selectedSources
    }, function() {
      settingsPanel.style.display = 'none';
      // Show a saved notification
      const notification = document.createElement('div');
      notification.textContent = 'Settings saved!';
      notification.style.color = 'green';
      notification.style.marginTop = '8px';
      notification.style.textAlign = 'center';
      document.querySelector('.settings').appendChild(notification);
      
      // Remove notification after 2 seconds
      setTimeout(() => {
        notification.remove();
      }, 2000);
    });
  });

  // Check the current page for facts
  checkPageButton.addEventListener('click', function() {
    // Show loader and hide previous results
    loader.style.display = 'block';
    results.style.display = 'none';
    summary.innerHTML = '';
    factsList.innerHTML = '';
    
    // Get the active tab
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
      const activeTab = tabs[0];
      
      // Send a message to the content script to analyze the page
      chrome.tabs.sendMessage(activeTab.id, {action: "analyze"}, function(response) {
        if (chrome.runtime.lastError) {
          // Handle error when content script is not loaded
          displayError("Could not connect to the page. Please refresh and try again.");
          return;
        }
        
        if (response && response.status === "analyzing") {
          // Start polling for results from the background script
          checkAnalysisStatus(activeTab.id);
        } else {
          displayError("Failed to start analysis. Please try again.");
        }
      });
    });
  });

  // Poll for analysis results
  function checkAnalysisStatus(tabId) {
    chrome.runtime.sendMessage({action: "getAnalysisStatus", tabId: tabId}, function(response) {
      if (response && response.status === "complete") {
        displayResults(response.results);
      } else if (response && response.status === "error") {
        displayError(response.message || "An error occurred during analysis.");
      } else {
        // Continue polling
        setTimeout(() => checkAnalysisStatus(tabId), 1000);
      }
    });
  }

  // Display error message
  function displayError(message) {
    loader.style.display = 'none';
    results.style.display = 'block';
    summary.innerHTML = `<div style="color: red;">${message}</div>`;
  }

  // Display analysis results
  function displayResults(data) {
    loader.style.display = 'none';
    results.style.display = 'block';
    
    // Count corrections
    const correctedFacts = data.facts.filter(fact => !fact.isTrue && fact.correction);
    
    // Display summary
    const accuracyPercentage = data.overallAccuracy;
    const accuracyClass = accuracyPercentage >= 80 ? 'success' : accuracyPercentage >= 60 ? 'warning' : 'danger';
    
    summary.innerHTML = `
      <h3>Article Analysis</h3>
      <p>Overall Accuracy: <span style="color: var(--${accuracyClass}-color); font-weight: bold;">${accuracyPercentage}%</span></p>
      <p>${data.summarySentence || ''}</p>
      ${correctedFacts.length > 0 ? 
        `<p><strong>${correctedFacts.length} numerical ${correctedFacts.length === 1 ? 'value' : 'values'} corrected</strong></p>` : 
        ''}
    `;
    
    // Display individual fact checks
    if (data.facts && data.facts.length > 0) {
      data.facts.forEach(fact => {
        const factItem = document.createElement('div');
        factItem.className = `fact-item fact-${fact.isTrue ? 'true' : 'uncertain' ? 'uncertain' : 'false'}`;
        
        let correctionHTML = '';
        if (!fact.isTrue && fact.correction) {
          correctionHTML = `
            <div class="correction">
              <span style="text-decoration: line-through; color: var(--danger-color);">${fact.correction.originalValue}</span>
              <span style="margin: 0 5px;">→</span>
              <span style="font-weight: bold; color: var(--success-color);">${fact.correction.correctedValue}</span>
            </div>
          `;
        }
        
        // Create enhanced source link with more detailed information
        let sourceLinkHTML = '';
        if (fact.sourceURL && fact.evidenceText) {
          sourceLinkHTML = `
            <div class="source-info">
              <div class="source-label">Source:</div>
              <div class="source-text">${fact.evidenceText}</div>
              <a href="${fact.sourceURL}" target="_blank" class="source-link">
                View Source Document
              </a>
            </div>
          `;
        }
        
        factItem.innerHTML = `
          <div class="claim"><strong>Claim:</strong> "${fact.claim}"</div>
          <div class="verdict">
            <strong>Verdict:</strong> 
            <span class="verdict-${fact.isTrue ? 'true' : 'false'}">${fact.isTrue ? 'True' : 'False'}</span>
          </div>
          ${correctionHTML}
          ${fact.explanation ? `<div class="explanation"><strong>Explanation:</strong> ${fact.explanation}</div>` : ''}
          ${sourceLinkHTML}
          <div class="confidence">
            <strong>Confidence:</strong> 
            <div class="confidence-bar">
              <div class="confidence-fill" style="width: ${fact.confidence}%; background-color: var(--${fact.confidence >= 70 ? 'success' : 'warning'}-color);"></div>
            </div>
            <span class="confidence-value">${fact.confidence}%</span>
          </div>
        `;
        
        factsList.appendChild(factItem);
      });
    } else {
      factsList.innerHTML = '<p>No specific facts were analyzed in this article.</p>';
    }
  }
});

// Add a test function for the SVM model
async function testSvmModel() {
  try {
    // Import and initialize the model
    const { AI } = await import('./ai_models.js');
    const svmModel = new AI.SVMFactCheckModel();
    
    // Log status
    console.log('Testing SVM model in browser environment...');
    
    // Test claims
    const testClaims = [
      "The unemployment rate is at a 50-year low of 3.5%.",
      "Global temperatures have risen by 1.1 degrees Celsius since pre-industrial times."
    ];
    
    // Wait for model initialization
    setTimeout(async () => {
      console.log('Running SVM model test...');
      
      for (const claim of testClaims) {
        const result = await svmModel.verifyClaim(claim);
        console.log(`SVM Result for "${claim}": ${result.isTrue ? 'TRUE' : 'FALSE'} (${result.confidence}% confidence)`);
      }
      
      console.log('SVM test completed');
    }, 5000);
    
    return true;
  } catch (error) {
    console.error('SVM test error:', error);
    return false;
  }
}

// Initialize the extension
function initializeExtension() {
  console.log('Initializing NewsFactChecker popup...');
  // Any initialization code can go here
  
  // Check for active tab analysis status
  chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
    if (tabs && tabs[0]) {
      checkAnalysisStatus(tabs[0].id);
    }
  });
}

// Run SVM test when the extension loads
document.addEventListener('DOMContentLoaded', () => {
  // Run the initialization code
  initializeExtension();
  
  // Also run the SVM test
  testSvmModel().then(success => {
    console.log('SVM test initialization:', success ? 'SUCCESS' : 'FAILED');
  });
});

/*



 It specifically focuses on identifying incorrect numerical information (like dollar amounts, percentages, statistics) and provides the correct values alongside the article text.

## Features

- **Numerical Fact Checking**: Automatically identifies and corrects incorrect numerical values in articles
- **Visual Highlighting**: Marks factually incorrect information directly on the webpage with corrections
- **Correction Footnotes**: Adds a summary of all corrections at the bottom of the article
- **Detailed Analysis**: Provides a summary of the article's overall factual accuracy
- **Evidence Sources**: Links to authoritative sources for each correction

## Examples of Corrections

The extension can identify and correct various types of numerical inaccuracies such as:

- A news claim states "$215 billion" but the true figure is "$320 billion"
- An article mentions "45%" when the actual percentage is "37%"
- A report says "500 people" when the correct number is "720 people"

## Installation

### From Chrome Web Store (Recommended)

1. Visit the Chrome Web Store page for NewsFactChecker (link will be available after publishing)
2. Click "Add to Chrome"
3. Confirm by clicking "Add extension"

### From Source (Development)

1. Clone this repository
2. Open Chrome and navigate to `chrome://extensions/`
3. Enable "Developer mode" (toggle in the top-right corner)
4. Click "Load unpacked" and select the directory where you cloned the repository
5. The extension should now be installed and ready to use

## Testing the Extension

To test the extension with the latest accuracy improvements:

1. Load the extension in developer mode as described above
2. Open the included `demo.html` file in Chrome
3. Click the NewsFactChecker icon in your browser toolbar
4. Click "Check This Article" to analyze the demo content
5. Observe how the extension:
   - Correctly highlights complete percentage values (fixes partial highlighting issue)
   - Provides specific links to authoritative sources for each correction
   - Shows evidence in tooltips when hovering over corrections
   - Displays a comprehensive summary of corrections with source links

## Usage

1. Navigate to any news article or blog post
2. Click the NewsFactChecker icon in your browser toolbar
3. Click "Check This Article" to begin analysis
4. View the overall accuracy rating and specific corrections in the popup
5. Factually incorrect statements will be highlighted directly on the webpage, with hover tooltips showing the correct values
6. A summary of all corrections is added at the bottom of the article
7. Click on any source link to see the evidence for a correction

## Publishing to Chrome Web Store

If you want to publish this extension to the Chrome Web Store, follow these steps:

### Step 1: Prepare Your Extension

1. Make sure you have proper icon files in the `icons` folder (16x16, 48x48, and 128x128 PNG files)
2. Update the manifest.json file with your details
3. Create a ZIP file of your extension folder (don't include .git directory or unnecessary files)

### Step 2: Create a Developer Account

1. Go to the [Chrome Web Store Developer Dashboard](https://chrome.google.com/webstore/devconsole/)
2. Sign in with your Google account
3. Pay the one-time $5 registration fee if you haven't already

### Step 3: Create a New Item

1. Click "New Item" in the developer dashboard
2. Upload your ZIP file
3. Fill in the required information:
   - Detailed description
   - At least 1-3 screenshots (1280x800 or 640x400)
   - Store icon (128x128)
   - Category (Productivity)
   - Language
4. Complete the privacy practices section
5. Submit for review

The review process typically takes a few business days. Once approved, your extension will be available in the Chrome Web Store.

## Technical Implementation

The extension uses pattern matching and heuristics to:

1. Identify claims containing numerical values (currency, percentages, statistics)
2. Determine if the values are accurate
3. Generate corrected values when inaccuracies are found
4. Present corrections in an easy-to-understand format
5. Link to authoritative sources for each correction

In a production environment, this would be enhanced with:
- Real fact databases for verification
- Machine learning for claim extraction and verification
- API connections to trusted data sources

## Future Improvements

- Connect to factual databases to perform real verification (rather than simulation)
- Implement machine learning for better claim extraction
- Add user feedback mechanisms to improve accuracy
- Support more languages and numerical formats
- Create a database of common factual errors and their corrections

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

*/