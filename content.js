// At the top of the file, add a context check
// This is a content script, so document should always be available
if (typeof document === 'undefined') {
  console.error('Content script running in a context where document is not available!');
} else {
  console.log('Content script running with document access available');
}

// Map to track which corrections have been added to the footnote
const addedCorrections = new Map();

// Use a flag to track if we've initialized yet
let factCheckerInitialized = false;

// Initialize when the page loads
if (typeof document !== 'undefined') {
  document.addEventListener('DOMContentLoaded', initializeFactChecker);
  
  // Also initialize immediately (for already-loaded pages)
  initializeFactChecker();
}

function initializeFactChecker() {
  // Only initialize once
  if (factCheckerInitialized) return;
  factCheckerInitialized = true;
  
  console.log('NewsFactChecker initialized on page:', window.location.href);
}

// Listen for messages from the popup
chrome.runtime.onMessage.addListener(function(message, sender, sendResponse) {
  console.log("Content script received message:", message);
  
  if (message.action === "analyze") {
    console.log("Starting analysis...");
    // Extract article content
    const article = extractArticleContent();
    console.log("Extracted article:", article);
    
    // Send article content to background script for processing
    chrome.runtime.sendMessage({
      action: "processArticle",
      article: article
    }, function(response) {
      if (chrome.runtime.lastError) {
        console.error('Error sending article to background:', chrome.runtime.lastError);
        sendResponse({status: "error", message: "Failed to start analysis"});
      } else {
        console.log("Article sent to background script");
        sendResponse({status: "analyzing"});
      }
    });
    
    return true; // Keep the message channel open for asynchronous response
  }
  
  if (message.action === "highlightFact") {
    highlightText(message.text, message.isFactual, message.correction, message.sourceURL, message.evidenceText);
    sendResponse({status: "highlighted"});
    return true;
  }
});

function extractArticleContent() {
  // Get article title
  let title = '';
  const titleElement = document.querySelector('h1');
  if (titleElement) {
    title = titleElement.textContent.trim();
  } else {
    title = document.title;
  }

  // Get article paragraphs - improved selector list
  const paragraphs = [];
  
  // Try multiple selectors to find article content
  const selectors = [
    '.article-content', 
    'article', 
    '[role="article"]', 
    '.post-content',
    '.entry-content',
    '.article-body',
    'main',
    '#content'
  ];
  
  let articleContent = null;
  
  // Try each selector until we find content
  for (const selector of selectors) {
    const element = document.querySelector(selector);
    if (element) {
      articleContent = element;
      break;
    }
  }
  
  // If no article container was found, fall back to body
  if (!articleContent) {
    articleContent = document.body;
  }
  
  // Find all paragraphs in the article content
  const pElements = articleContent.querySelectorAll('p');
  for (const p of pElements) {
    // Skip if paragraph is too short or appears to be a caption/metadata
    if (p.textContent.trim().length > 20 && !p.closest('figure, figcaption, nav, header, footer')) {
      paragraphs.push(p.textContent.trim());
    }
  }
  
  // If no paragraphs were found, use all text nodes
  if (paragraphs.length === 0) {
    const textNodes = [];
    const walker = document.createTreeWalker(
      articleContent,
      NodeFilter.SHOW_TEXT,
      null,
      false
    );
    
    let node;
    while (node = walker.nextNode()) {
      const text = node.nodeValue.trim();
      if (text.length > 20) {
        textNodes.push(text);
      }
    }
    
    // Group text nodes into paragraph-sized chunks
    let currentParagraph = '';
    for (const text of textNodes) {
      currentParagraph += text + ' ';
      if (currentParagraph.length > 100) {
        paragraphs.push(currentParagraph.trim());
        currentParagraph = '';
      }
    }
    
    // Add the last paragraph if it exists
    if (currentParagraph.trim().length > 0) {
      paragraphs.push(currentParagraph.trim());
    }
  }
  
  console.log("Extracted paragraphs:", paragraphs.length);
  
  return {
    title,
    url: window.location.href,
    paragraphs
  };
}

// Function to highlight text and add corrections if needed
function highlightText(text, isFactual, correction, sourceURL, evidenceText) {
  if (!text) return; // Skip if text is empty
  
  // Clear the correction map for new analysis
  if (document.getElementById('news-fact-checker-corrections')) {
    document.getElementById('news-fact-checker-corrections').remove();
    addedCorrections.clear();
  }
  
  // Find specific numerical values to highlight if we have a correction
  let valuesToHighlight = [];
  
  if (correction) {
    // Extract the original value to precisely target it
    valuesToHighlight.push({
      value: correction.originalValue,
      replacement: correction.correctedValue
    });
  } else {
    // If no correction provided, still look for numerical values to highlight
    const numericalValues = extractNumericalValues(text);
    valuesToHighlight = numericalValues.map(val => ({
      value: val,
      replacement: null  // No replacement for factually correct values
    }));
  }
  
  const textNodes = [];
  
  // Find all text nodes in the document
  const walker = document.createTreeWalker(
    document.body,
    NodeFilter.SHOW_TEXT,
    null,
    false
  );
  
  let node;
  while (node = walker.nextNode()) {
    // Check for the full text match
    if (node.nodeValue.includes(text)) {
      textNodes.push({node, matchType: 'full', value: text});
    }
    // Also check for just numerical values  
    else if (valuesToHighlight.length > 0) {
      for (const {value} of valuesToHighlight) {
        if (node.nodeValue.includes(value)) {
          textNodes.push({node, matchType: 'value', value});
        }
      }
    }
  }
  
  console.log(`Found ${textNodes.length} text nodes to highlight for: ${text}`);
  
  // Highlight each occurrence
  textNodes.forEach(({node, matchType, value}) => {
    const parent = node.parentNode;
    const content = node.nodeValue;
    
    // Skip if this node is already inside a fact-checked element (prevent double highlighting)
    if (parent.dataset.factChecked === 'true' || parent.closest('[data-fact-checked="true"]')) {
      return;
    }
    
    // If we're matching a specific numerical value
    if (matchType === 'value' && correction) {
      // More precise highlighting of just the numerical value
      const parts = content.split(value);
      
      if (parts.length > 1) {
        // Create fragment to hold the new nodes
        const fragment = document.createDocumentFragment();
        
        // Add the first part
        if (parts[0]) {
          fragment.appendChild(document.createTextNode(parts[0]));
        }
        
        // Create the highlighted element for just the numerical value
        const highlightedSpan = document.createElement('span');
        highlightedSpan.dataset.factChecked = 'true';
        highlightedSpan.classList.add('fact-correction');
        
        // Style for incorrect value
        highlightedSpan.style.backgroundColor = 'rgba(244, 67, 54, 0.2)';
        highlightedSpan.style.border = '1px solid rgba(244, 67, 54, 0.5)';
        highlightedSpan.style.borderRadius = '2px';
        highlightedSpan.style.position = 'relative';
        highlightedSpan.style.textDecoration = 'line-through';
        
        // Create a more detailed tooltip with source information
        const tooltipTitle = sourceURL ? 
          `Correction: ${value} → ${correction.correctedValue}. Source: ${evidenceText}` :
          `Correction: ${value} → ${correction.correctedValue}`;
        
        highlightedSpan.title = tooltipTitle;
        
        // Add original value with strikethrough
        highlightedSpan.textContent = value;
        
        // Add enhanced tooltip that appears on hover
        const tooltip = document.createElement('span');
        tooltip.style.position = 'absolute';
        tooltip.style.top = '-24px';
        tooltip.style.left = '0';
        tooltip.style.backgroundColor = '#fff';
        tooltip.style.border = '1px solid #ccc';
        tooltip.style.padding = '6px 10px';
        tooltip.style.borderRadius = '4px';
        tooltip.style.boxShadow = '0 2px 8px rgba(0,0,0,0.15)';
        tooltip.style.zIndex = '1000';
        tooltip.style.fontSize = '12px';
        tooltip.style.color = '#333';
        tooltip.style.display = 'none';
        tooltip.style.maxWidth = '300px';
        tooltip.style.lineHeight = '1.5';
        tooltip.style.whiteSpace = 'normal';
        tooltip.style.fontFamily = 'Arial, sans-serif';
        
        // Add corrected value with better formatting
        const correctionText = document.createElement('div');
        correctionText.style.fontWeight = 'bold';
        correctionText.style.color = '#4CAF50';
        correctionText.style.marginBottom = '4px';
        correctionText.innerHTML = `Correct value: <span style="color:#4CAF50">${correction.correctedValue}</span>`;
        tooltip.appendChild(correctionText);
        
        // Add source information with more detail
        if (sourceURL && evidenceText) {
          const sourceInfo = document.createElement('div');
          sourceInfo.style.fontSize = '11px';
          sourceInfo.style.color = '#666';
          sourceInfo.style.marginBottom = '6px';
          sourceInfo.textContent = `Source: ${evidenceText}`;
          tooltip.appendChild(sourceInfo);
          
          const sourceLink = document.createElement('a');
          sourceLink.href = sourceURL;
          sourceLink.target = '_blank';
          sourceLink.textContent = 'View Source Document';
          sourceLink.style.display = 'block';
          sourceLink.style.marginTop = '4px';
          sourceLink.style.color = '#2196F3';
          sourceLink.style.textDecoration = 'none';
          sourceLink.style.fontWeight = 'bold';
          sourceLink.style.fontSize = '11px';
          tooltip.appendChild(sourceLink);
        }
        
        highlightedSpan.appendChild(tooltip);
        
        // Show tooltip on hover
        highlightedSpan.addEventListener('mouseenter', () => {
          tooltip.style.display = 'block';
        });
        
        highlightedSpan.addEventListener('mouseleave', () => {
          tooltip.style.display = 'none';
        });
        
        fragment.appendChild(highlightedSpan);
        
        // Add the remaining parts
        for (let i = 1; i < parts.length; i++) {
          if (parts[i]) {
            fragment.appendChild(document.createTextNode(parts[i]));
          }
        }
        
        // Replace the original text node with our highlighted version
        parent.replaceChild(fragment, node);
        
        // Add to footnote
        if (correction && !addedCorrections.has(value)) {
          addFootnoteCorrection(text, correction, sourceURL, evidenceText);
          addedCorrections.set(value, true);
        }
      }
    } else if (matchType === 'full') {
      // Highlight the entire claim
      const parts = content.split(text);
      
      if (parts.length > 1) {
        // Create fragment to hold the new nodes
        const fragment = document.createDocumentFragment();
        
        // Add the first part
        if (parts[0]) {
          fragment.appendChild(document.createTextNode(parts[0]));
        }
        
        // Create the highlighted element
        const highlightedSpan = document.createElement('span');
        highlightedSpan.dataset.factChecked = 'true';
        
        if (isFactual) {
          // Factually correct
          highlightedSpan.textContent = text;
          highlightedSpan.style.backgroundColor = 'rgba(76, 175, 80, 0.1)';
          highlightedSpan.style.border = '1px solid rgba(76, 175, 80, 0.3)';
          highlightedSpan.style.borderRadius = '2px';
          highlightedSpan.title = 'Factually correct';
        } else {
          // Factually incorrect
          if (correction) {
            // Create a special annotation for numerical corrections
            highlightedSpan.classList.add('fact-correction');
            highlightedSpan.style.backgroundColor = 'rgba(244, 67, 54, 0.2)';
            highlightedSpan.style.border = '1px solid rgba(244, 67, 54, 0.5)';
            highlightedSpan.style.borderRadius = '2px';
            
            // Enhanced tooltip with source information
            const tooltipTitle = sourceURL ? 
              `Contains incorrect information: ${correction.originalValue} should be ${correction.correctedValue}. Source: ${evidenceText}` :
              `Contains incorrect information. See footnote for details.`;
            
            highlightedSpan.title = tooltipTitle;
            highlightedSpan.textContent = text;
            
            // Add to footnote
            if (!addedCorrections.has(correction.originalValue)) {
              addFootnoteCorrection(text, correction, sourceURL, evidenceText);
              addedCorrections.set(correction.originalValue, true);
            }
          } else {
            // Regular highlighting for non-numerical incorrect facts
            highlightedSpan.textContent = text;
            highlightedSpan.style.backgroundColor = 'rgba(244, 67, 54, 0.2)';
            highlightedSpan.style.border = '1px solid rgba(244, 67, 54, 0.5)';
            highlightedSpan.style.borderRadius = '2px';
            highlightedSpan.title = 'Factually incorrect';
          }
        }
        
        fragment.appendChild(highlightedSpan);
        
        // Add the remaining parts
        for (let i = 1; i < parts.length; i++) {
          if (parts[i]) {
            fragment.appendChild(document.createTextNode(parts[i]));
          }
        }
        
        // Replace the original text node with our highlighted version
        parent.replaceChild(fragment, node);
      }
    }
  });
}

// Function to extract numerical values from text
function extractNumericalValues(text) {
  const values = [];
  
  // Comprehensive regex for finding numerical values
  const patterns = [
    /\$\d+(?:,\d+)*(?:\.\d+)?\s*(?:billion|million|trillion|thousand)?/g,  // Currency with optional units
    /\b\d+(?:\.\d+)?%\b/g,  // Percentages
    /\b\d+(?:,\d+)*(?:\.\d+)?\s+(?:people|individuals|users|customers|years|months|days)\b/g,  // Numbers with units
  ];
  
  // Find all matches
  patterns.forEach(pattern => {
    let match;
    while ((match = pattern.exec(text)) !== null) {
      values.push(match[0]);
    }
  });
  
  return values;
}

// Function to add a footnote with the correction at the end of the article
function addFootnoteCorrection(text, correction, sourceURL, evidenceText) {
  // Check if we already have a corrections footnote section
  let correctionsFootnote = document.getElementById('news-fact-checker-corrections');
  
  if (!correctionsFootnote) {
    // Create the corrections footnote section
    correctionsFootnote = document.createElement('div');
    correctionsFootnote.id = 'news-fact-checker-corrections';
    correctionsFootnote.style.margin = '30px 0';
    correctionsFootnote.style.padding = '20px';
    correctionsFootnote.style.backgroundColor = '#f8f9fa';
    correctionsFootnote.style.border = '1px solid #e0e0e0';
    correctionsFootnote.style.borderRadius = '8px';
    correctionsFootnote.style.boxShadow = '0 2px 10px rgba(0,0,0,0.05)';
    correctionsFootnote.style.fontFamily = 'Arial, sans-serif';
    
    const footnoteTitle = document.createElement('h3');
    footnoteTitle.textContent = 'Fact Checker Corrections';
    footnoteTitle.style.marginTop = '0';
    footnoteTitle.style.marginBottom = '15px';
    footnoteTitle.style.color = '#d32f2f';
    footnoteTitle.style.fontSize = '20px';
    correctionsFootnote.appendChild(footnoteTitle);
    
    // Add a description
    const description = document.createElement('p');
    description.textContent = 'The following corrections are based on verified data from authoritative sources:';
    description.style.marginBottom = '15px';
    description.style.fontSize = '14px';
    description.style.color = '#555';
    correctionsFootnote.appendChild(description);
    
    const footnoteList = document.createElement('ul');
    footnoteList.id = 'corrections-list';
    footnoteList.style.paddingLeft = '20px';
    correctionsFootnote.appendChild(footnoteList);
    
    // Find a good place to insert the footnote (end of article or body)
    const articleContainers = [
      document.querySelector('article'),
      document.querySelector('.article'),
      document.querySelector('.post-content'),
      document.querySelector('.entry-content'),
      document.querySelector('.content'),
      document.querySelector('main'),
      document.querySelector('#content')
    ];
    
    let container = null;
    for (const candidate of articleContainers) {
      if (candidate) {
        container = candidate;
        break;
      }
    }
    
    if (!container) {
      container = document.body;
    }
    
    container.appendChild(correctionsFootnote);
  }
  
  // Check if we've already added this correction
  const correctionsList = document.getElementById('corrections-list');
  
  // Skip if we've already added this correction
  if (addedCorrections.has(correction.originalValue)) {
    return;
  }
  
  const correctionItem = document.createElement('li');
  correctionItem.style.margin = '12px 0';
  correctionItem.style.lineHeight = '1.6';
  correctionItem.style.fontSize = '14px';
  
  // Create a short snippet of the claim (first 80 chars)
  const shortClaim = text.length > 80 ? text.substring(0, 77) + '...' : text;
  
  // Create the HTML for the correction with enhanced source information
  let correctionHTML = `
    <div style="margin-bottom: 8px;">
      <span style="font-weight: bold;">Claim:</span> "${shortClaim}"
    </div>
    <div style="display: flex; align-items: center; margin-bottom: 6px;">
      <span style="text-decoration: line-through; color: #f44336; margin-right: 8px;">${correction.originalValue}</span>
      <span style="color: #757575; margin-right: 8px;">→</span>
      <span style="font-weight: bold; color: #4CAF50;">${correction.correctedValue}</span>
    </div>
  `;
  
  if (sourceURL && evidenceText) {
    correctionHTML += `
      <div style="margin-top: 6px; font-size: 13px;">
        <span style="color: #555; font-weight: bold;">Source:</span> 
        <span style="color: #333;">${evidenceText}</span><br>
        <a href="${sourceURL}" target="_blank" style="color: #1976D2; text-decoration: none; font-weight: 500; display: inline-block; margin-top: 4px; border-bottom: 1px solid #1976D2;">View Source Document</a>
      </div>
    `;
  }
  
  correctionItem.innerHTML = correctionHTML;
  correctionsList.appendChild(correctionItem);
} 