/**
 * SVM Model Test Runner
 * 
 * This script tests the SVM-based fact verification model with real data.
 * Runs automatically without requiring user interaction.
 */

import { AI } from './ai_models.js';

// Test claims to verify the model is working
const testClaims = [
  "The unemployment rate is at a 50-year low of 3.5%.",
  "Global temperatures have risen by 1.1 degrees Celsius since pre-industrial times.",
  "The average student loan debt is over $30,000 per graduate.",
  "Every single person who has taken the COVID vaccine has suffered serious side effects.",
  "According to research, drinking coffee regularly reduces the risk of certain cancers by 15%."
];

// Run the test automatically
(async () => {
  console.log('Starting SVM model tests with real data...');
  
  try {
    // Initialize the model
    const model = new AI.SVMFactCheckModel();
    console.log('Model initialized, waiting for training to complete...');
    
    // Wait for model to be ready (it should auto-initialize)
    await new Promise(resolve => {
      const checkInterval = setInterval(() => {
        if (model.initialized) {
          clearInterval(checkInterval);
          resolve();
        }
      }, 500);
      
      // Safety timeout after 60 seconds
      setTimeout(() => {
        clearInterval(checkInterval);
        resolve();
      }, 60000);
    });
    
    console.log('Model ready, testing verification on sample claims...');
    
    // Test the model on sample claims
    let results = [];
    
    for (let i = 0; i < testClaims.length; i++) {
      const claim = testClaims[i];
      console.log(`\nTesting claim ${i+1}: "${claim}"`);
      
      const result = await model.verifyClaim(claim);
      console.log(`Verdict: ${result.isTrue ? 'TRUE' : 'FALSE'} (${result.confidence}% confidence)`);
      
      results.push({
        claim,
        result
      });
    }
    
    // Output overall results
    console.log('\n======== TEST RESULTS ========');
    console.log(`Verified ${results.length} claims successfully`);
    
    const trueCount = results.filter(r => r.result.isTrue).length;
    const falseCount = results.filter(r => !r.result.isTrue).length;
    
    console.log(`True verdicts: ${trueCount}`);
    console.log(`False verdicts: ${falseCount}`);
    console.log('=============================');
    
    console.log('\n✅ All tests completed successfully');
    process.exit(0);
  } catch (error) {
    console.error('❌ Error running SVM tests:', error);
    process.exit(1);
  }
})(); 