<video width="640" height="360" controls>
  <source src="vid/demo.mov" type="video/mov">
</video>




[Watch the demo video](vid/demo.mov)
# NewsFactChecker

A Chrome extension that identifies and corrects factual inaccuracies in online articles, with a focus on numerical claims.

## Overview

NewsFactChecker is a browser extension that analyzes online articles for factual accuracy. It specifically focuses on identifying incorrect numerical information (like dollar amounts, percentages, statistics) and provides the correct values alongside the article text.

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

Contributions are welcome! Please open an issue or submit a pull request.




Check if validd
Write tests first, then the code, then run the tests and update the code until tests pass.

Please add logs to the code to get better visibility into what is going on so we can find the fix. I'll run the code and feed you the logs results


Fundamental Principles

1. Write clean, simple, readable code  
2. Implement features in the simplest possible way  
3. Keep files small and focused (<200 lines)  
4. Test after every meaningful change  
5. Focus on core functionality before optimization  
6. Use clear, consistent naming  
7. Think thoroughly before coding. Write 2–3 reasoning paragraphs.  
8. ALWAYS write simple, clean, and modular code.  
9. Use clear and easy-to-understand language. Write in short sentences.

Error Fixing

- DO NOT JUMP TO CONCLUSIONS! Consider multiple possible causes before deciding.  
- Explain the problem in plain English.  
- Make minimal necessary changes, changing as few lines of code as possible.  
- In case of strange errors, ask the user to perform a Perplexity Web search to get the most up-to-date information.

Debugging

- Briefly review how each feature works by telling the user how to test it.  
- DO NOT write complicated and confusing code. Opt for the simplest approach.  
- When not sure what to do, tell the user to perform a web search.

Building

- ALWAYS try to add more helpful and explanatory comments into your code.  
- NEVER delete old comments — unless they are obviously wrong or obsolete.  
- Include LOTS of explanatory comments in your code. ALWAYS be well-labeled.  
- Document all changes and their reasoning in the comments you write.  
- When writing comments, use clear and easy-to-understand language.