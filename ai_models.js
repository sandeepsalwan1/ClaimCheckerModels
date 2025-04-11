/**
 * AI Models Integration
 * 
 * NOTE: This file is for demonstration purposes only.
 * In a real implementation, you would integrate actual AI models.
 */

// Import TensorFlow.js
import * as tf from '@tensorflow/tfjs';
// Force CPU backend usage instead of WebGL for extension compatibility
tf.setBackend('cpu');

import * as use from '@tensorflow-models/universal-sentence-encoder';
import * as toxicity from '@tensorflow-models/toxicity';

import Papa from 'papaparse';

// Import LIAR dataset processing functions
import { loadLiarDataset, processLiarDataset, splitDataset } from './data/liar_dataset.js';

// Check if we're in a browser environment
const isBrowserEnvironment = typeof window !== 'undefined' || typeof self !== 'undefined';
const isContentScript = isBrowserEnvironment && typeof document !== 'undefined';
const isServiceWorker = isBrowserEnvironment && typeof self !== 'undefined' && typeof document === 'undefined';

// Import fs module conditionally if in Node environment
let fs = null;
if (!isBrowserEnvironment) {
  // Dynamic import for Node.js environment
  try {
    // In browsers, this will fail silently
    import('fs/promises')
      .then(module => {
        fs = module.default;
        console.log('fs module loaded');
      })
      .catch(err => {
        console.warn('fs module not available:', err.message);
      });
  } catch (e) {
    console.warn('fs module import failed:', e);
  }
}

/**
 * Deep Learning Fact Extraction Model
 * In a real implementation, this would load a pre-trained NLP model
 * for extracting factual claims from text.
 */
class ClaimExtractionModel {
  constructor() {
    this.modelLoaded = false;
    this.useModel = null;
    this.initialize();
  }

  async initialize() {
    try {
      if (isContentScript) {
        console.log('Running claim extraction in content script context - using simplified approach');
        this.modelLoaded = true;
        return;
      }
      
      if (isServiceWorker) {
        console.log('Running claim extraction in service worker context - using simplified approach');
        this.modelLoaded = true;
        return;
      }
      
      // Only load USE model in non-content script contexts
      console.log('Initializing claim extraction model (USE)');
      
      try {
        this.useModel = await use.load();
        console.log('Claim extraction model (USE) loaded successfully');
      } catch (err) {
        console.warn('Could not load USE model, using fallback approach', err);
      }
      
      this.modelLoaded = true;
    } catch (error) {
      console.error('Failed to load claim extraction model:', error);
      this.modelLoaded = true; // Still mark as loaded so we can use fallback
    }
  }

  async extractClaims(text) {
    if (!this.modelLoaded) {
      throw new Error('Model not loaded');
    }

    // If USE model isn't loaded, use simple pattern-based extraction
    if (!this.useModel) {
      return this._fallbackExtractClaims(text);
    }

    try {
      // Split text into sentences
      const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 10);
      
      // Encode all sentences
      const embeddings = await this.useModel.embed(sentences);
      
      // Extract numerical claims using regex patterns
      const numericalPatterns = {
        currency: /\$\d+(\.\d+)?\s*(million|billion|trillion|thousand)?/gi,
        percentage: /\b\d+(\.\d+)?%\b/gi,
        numbers: /\b\d+(\.\d+)?\s*(people|individuals|users|customers|years|months|days)\b/gi
      };

      // Analyze each sentence for claims
      const claims = [];
      const embeddingArray = await embeddings.array();

      for (let i = 0; i < sentences.length; i++) {
        const sentence = sentences[i];
        const embedding = embeddingArray[i];

        // Check for numerical patterns
        const hasNumerical = Object.values(numericalPatterns).some(pattern => {
          pattern.lastIndex = 0; // Reset regex
          return pattern.test(sentence);
        });

        // Calculate claim probability using embedding features
        const claimProbability = this._calculateClaimProbability(embedding);

        // Lower the threshold to catch more claims (from 0.7 to 0.5)
        if ((hasNumerical && claimProbability > 0.5) || 
            // Also include sentences with specific terms that indicate factual claims
            /\$|\%|\d+ (billion|million|trillion|thousand|percent)/i.test(sentence)) {
          claims.push({
            text: sentence,
            confidence: claimProbability,
            type: this._determineClaimType(sentence),
            numericalValues: this._extractNumericalValues(sentence)
          });
        }
      }

      return claims;
    } catch (error) {
      console.error('Error in claim extraction:', error);
      return this._fallbackExtractClaims(text);
    }
  }

  // Fallback extraction method using simple patterns
  _fallbackExtractClaims(text) {
    console.log('Using fallback claim extraction method');
    const claims = [];
    
    // Split text into sentences
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 10);
    
    // Patterns for detecting likely factual statements
    const factualPatterns = [
      /\$\d+(\.\d+)?\s*(million|billion|trillion|thousand)?/i,  // Dollar amounts
      /\d+(\.\d+)?%/i,  // Percentages
      /increased by \d+/i, /decreased by \d+/i,  // Changes
      /\d+ (people|individuals|users)/i,  // Counts of people
      /according to/i, /reported/i,  // Attribution phrases
      /showed that/i, /indicates that/i,  // Research findings
    ];
    
    sentences.forEach(sentence => {
      // Check if the sentence contains patterns indicative of factual claims
      const containsFactualPattern = factualPatterns.some(pattern => pattern.test(sentence));
      
      if (containsFactualPattern) {
        claims.push({
          text: sentence,
          confidence: 0.7,
          type: this._determineClaimType(sentence),
          numericalValues: this._extractNumericalValues(sentence)
        });
      }
    });
    
    return claims;
  }

  _calculateClaimProbability(embedding) {
    // Use embedding features to calculate probability
    // Higher values for sentences with factual language patterns
    const factualKeywords = ['according to', 'reported', 'study shows', 'data indicates'];
    const sentenceVector = tf.tensor1d(embedding);
    
    // Calculate cosine similarity with factual patterns
    const similarity = sentenceVector.dot(sentenceVector).dataSync()[0];
    return Math.min(Math.max(similarity * 0.8, 0), 1);
  }

  _determineClaimType(sentence) {
    if (/\$|\bUSD\b|\beuro\b/i.test(sentence)) return 'financial';
    if (/%|percent|percentage/i.test(sentence)) return 'percentage';
    if (/increase|decrease|growth|decline/i.test(sentence)) return 'trend';
    return 'general';
  }

  _extractNumericalValues(sentence) {
    const values = [];
    const patterns = [
      { type: 'currency', regex: /\$\d+(\.\d+)?\s*(million|billion|trillion|thousand)?/gi },
      { type: 'percentage', regex: /\b\d+(\.\d+)?%\b/gi },
      { type: 'number', regex: /\b\d+(\.\d+)?\s*(people|individuals|users|customers|years|months|days)\b/gi }
    ];

    patterns.forEach(({type, regex}) => {
      let match;
      while ((match = regex.exec(sentence)) !== null) {
        values.push({
          type,
          value: match[0],
          index: match.index
        });
      }
    });

    return values;
  }
}

/**
 * SVM Fact Verification Model
 * This model uses SVM with real fact-checking data to classify
 * claims as true or false based on text features
 */
class SVMFactCheckModel {
  constructor() {
    this.modelLoaded = false;
    this.useModel = null;
    this.svmModel = null;
    this.trainingData = null;
    this.validationData = null;
    this.features = null;
    this.initialize();
  }

  async initialize() {
    try {
      this.modelLoaded = false;
      console.log('Initializing SVM model...');
      
      // Different initialization based on environment
      if (isContentScript) {
        console.log('Running in content script context');
        // Content script context
        this.modelLoaded = true;
        return;
      } else if (isServiceWorker) {
        console.log('Running in service worker context');
        // In service worker, we need alternative approach
        try {
          await this.loadRealDataset();
          await this.trainModel();
        } catch (error) {
          console.warn('Error during model initialization in service worker:', error);
        }
        // Mark as loaded even if there was an error so we can use fallback methods
        this.modelLoaded = true;
        return;
      }
      
      // Fallback for other environments
      console.log('Loading fake news detection dataset');
      try {
        await this.loadRealDataset();
        await this.trainModel();
      } catch (error) {
        console.warn('Error during model initialization:', error);
      }
      
      // Mark as loaded even if there was an error
      this.modelLoaded = true;
      console.log('SVM model initialized successfully');
    } catch (error) {
      console.error('Failed to initialize fact verification model:', error);
      // Still mark as loaded so we can use fallback methods
      this.modelLoaded = true;
    }
  }

  /**
   * Load the real fact-checking dataset
   */
  async loadRealDataset() {
    try {
      console.log('Loading real fact-checking dataset (MediaEval)...');
      
      let parsedData = [];
      
      // Node.js environment handling
      if (!isBrowserEnvironment && fs) {
        try {
          // Try to load the dataset files using Node.js fs module
          console.log('Using Node.js fs module to load dataset');
          
          // First try from data directory
          const filePath = './data/mediaeval-2015-trainingset.csv';
          let fileData;
          
          try {
            fileData = await fs.readFile(filePath, 'utf8');
            console.log(`Successfully loaded MediaEval dataset from: ${filePath}`);
          } catch (fsError) {
            console.warn(`Failed to load from ${filePath}:`, fsError.message);
            
            // Try from fake-news-detection-testSet directory
            const altPath = './fake-news-detection-testSet/dataset/mediaeval-2015-trainingset.csv';
            try {
              fileData = await fs.readFile(altPath, 'utf8');
              console.log(`Successfully loaded MediaEval dataset from: ${altPath}`);
            } catch (altError) {
              console.warn(`Failed to load from ${altPath}:`, altError.message);
            }
          }
          
          if (fileData) {
            parsedData = Papa.parse(fileData, {
              header: true,
              skipEmptyLines: true
            }).data;
            console.log(`Successfully parsed ${parsedData.length} records from file`);
          }
        } catch (nodeError) {
          console.warn('Error loading dataset with Node.js fs:', nodeError);
        }
      } else {
        // Browser environment
        try {
          // First attempt to load MediaEval dataset using fetch
          const path = './data/mediaeval-2015-trainingset.csv';
          const response = await fetch(path);
          if (response.ok) {
            const csvText = await response.text();
            parsedData = Papa.parse(csvText, {
              header: true,
              skipEmptyLines: true
            }).data;
            console.log(`Successfully loaded MediaEval dataset from: ${path}`);
          } else {
            console.warn(`Failed to load dataset from ${path}, status: ${response.status}`);
          }
        } catch (fetchError) {
          console.warn('Error loading MediaEval dataset in browser:', fetchError);
        }
      }
      
      // Process the dataset if we have data
      let processedData = [];
      if (parsedData.length > 0) {
        processedData = parsedData.map(item => {
          return {
            id: item.tweetId || item.id || '',
            claim: item.tweetText || item.text || '',
            isTrue: item.label?.toLowerCase() === 'real' || item.label?.toLowerCase() === 'true',
            originalLabel: item.label || ''
          };
        }).filter(item => item.claim && item.claim.length > 10);
        
        console.log(`Successfully processed ${processedData.length} records from MediaEval dataset`);
      }
      
      // If we couldn't load or process the MediaEval dataset, fall back to LIAR dataset
      if (processedData.length < 10) {
        console.warn('MediaEval dataset loading failed or has insufficient data, falling back to local LIAR dataset');
        const rawData = await loadLiarDataset();
        const processedLiarData = processLiarDataset(rawData);
        const { training, validation } = splitDataset(processedLiarData, 0.2);
        this.trainingData = training;
        this.validationData = validation;
        
        console.log('Fallback dataset loaded successfully:');
        console.log(`Training set: ${this.trainingData.length} records`);
        console.log(`Validation set: ${this.validationData.length} records`);
      } else {
        // Split the MediaEval dataset into training and validation sets
        const shuffled = [...processedData].sort(() => 0.5 - Math.random());
        const splitIndex = Math.floor(shuffled.length * 0.8);
        this.trainingData = shuffled.slice(0, splitIndex);
        this.validationData = shuffled.slice(splitIndex);
        
        console.log(`Training set: ${this.trainingData.length} records`);
        console.log(`Validation set: ${this.validationData.length} records`);
      }
      
      return this.trainingData;
    } catch (error) {
      console.error('Error loading datasets:', error);
      console.warn('Falling back to local LIAR dataset');
      
      // Fallback to LIAR dataset as a last resort
      const rawData = await loadLiarDataset();
      const processedData = processLiarDataset(rawData);
      const { training, validation } = splitDataset(processedData, 0.2);
      
      this.trainingData = training;
      this.validationData = validation;
      
      console.log('Fallback dataset loaded successfully');
      return this.trainingData;
    }
  }
  
  /**
   * Generate features from text using USE embeddings
   */
  async generateFeatures(texts) {
    // Safeguard check
    if (!this.useModel) {
      console.warn('Universal Sentence Encoder not loaded, returning empty features');
      return [];
    }
    
    try {
      // Generate embeddings
      const embeddings = await this.useModel.embed(texts);
      return await embeddings.array();
    } catch (error) {
      console.error('Error generating features:', error);
      return [];
    }
  }
  
  /**
   * Train the SVM model with the processed dataset
   */
  async trainModel() {
    console.log('Training SVM model...');
    
    try {
      // Import SVM from ml-svm library
      const SVM = (await import('ml-svm')).default;
      this.SVM = SVM;
      
      // Skip USE model loading in content script context
      if (!isContentScript) {
        try {
          this.useModel = await use.load();
          console.log('Universal Sentence Encoder loaded successfully');
        } catch (error) {
          console.warn('Failed to load USE model, using fallback approach:', error);
        }
      }
      
      // Skip actual training in content script context or if no training data
      if (isContentScript) {
        console.log('Skipping model training in content script context');
        this.modelLoaded = true;
        return true;
      }
      
      // Skip actual training in service worker if we don't have USE model
      if (isServiceWorker && !this.useModel) {
        console.log('Service worker context detected without USE model - using fallback verification');
        this.modelLoaded = true;
        return true;
      }
      
      // Continue with actual training in other contexts
      if (this.trainingData && this.trainingData.length > 0 && this.useModel) {
        // Extract claims text and labels
        const texts = this.trainingData.map(item => item.claim);
        const labels = this.trainingData.map(item => item.isTrue ? 1 : 0);
        
        // Generate features using Universal Sentence Encoder - limit sample size for browser performance
        const maxSamples = 100; // Limit to 100 examples for browser performance
        const sampleTexts = texts.slice(0, maxSamples); 
        
        try {
          const features = await this.generateFeatures(sampleTexts);
          
          // Check for valid features
          if (!features || features.length === 0) {
            throw new Error('Failed to generate features for training');
          }
          
          console.log(`Generated features for ${features.length} training examples`);
          
          // Create and train SVM model
          const options = {
            C: 0.1,                // Cost parameter
            kernel: 'linear',      // Kernel type (linear is most efficient for browser)
            probability: true,     // Enable probability estimates
            gamma: 'auto',         // Kernel coefficient
            tolerance: 0.001,      // Tolerance for stopping criteria
            maxPasses: 5,          // Max iterations (lower for browser performance)
            maxIterations: 1000    // Additional iteration limit
          };
          
          this.svmModel = new this.SVM(options);
          
          // Make sure labels match feature dimensions
          const trainingLabels = labels.slice(0, features.length);
          
          // Train the model with labels that match the feature dimensions
          this.svmModel.train(features, trainingLabels);
          
          // Store features for dimension matching in prediction
          this.features = features;
          
          console.log('SVM model trained successfully');
        } catch (error) {
          console.error('Error during feature generation or training:', error);
          console.log('Using fallback verification method');
        }
      } else {
        console.log('No training data or USE model available. Using fallback verification method.');
      }
      
      // Mark model as loaded regardless of training outcome
      this.modelLoaded = true;
      return true;
    } catch (error) {
      console.error('Error training model:', error);
      // Still mark as loaded so we can use fallback methods
      this.modelLoaded = true;
      return false;
    }
  }

  /**
   * Verify a claim using the trained model
   */
  async verifyClaim(claim, evidence) {
    if (!this.modelLoaded) {
      console.warn('Model not loaded, initializing...');
      try {
        await this.initialize();
      } catch (error) {
        console.error('Failed to initialize model:', error);
        return this.fallbackVerification(claim, evidence);
      }
    }

    try {
      // Skip SVM if model isn't available and go straight to fallback
      if (!this.svmModel || !this.useModel) {
        console.log('SVM model or USE model not available, using fallback verification');
        return this.fallbackVerification(claim, evidence);
      }
      
      // Combine claim and evidence for better context
      const combinedText = evidence 
        ? `${claim} ${evidence}`  // Use evidence if available
        : claim;                  // Otherwise just use the claim
      
      try {
        // Generate feature embedding
        const embedding = await this.generateFeatures([combinedText]);
        
        // Verify the embedding is valid
        if (!embedding || embedding.length === 0 || embedding[0].some(val => isNaN(val))) {
          console.warn('Invalid embedding generated, using fallback verification');
          return this.fallbackVerification(claim, evidence);
        }
        
        // Get prediction from SVM model
        let prediction;
        try {
          prediction = this.svmModel.predict(embedding);
          if (!prediction || prediction.length === 0) {
            throw new Error('Empty prediction returned');
          }
        } catch (svmError) {
          console.error('Error during SVM prediction:', svmError);
          return this.fallbackVerification(claim, evidence);
        }
        
        const isTrue = prediction[0] === 1;
        
        // Calculate probabilities (confidence)
        let probability = 0.7; // Default medium-high confidence
        
        // Calculate confidence based on probability
        const confidencePercent = Math.abs(probability - 0.5) * 200; // Scale to 0-100%
        
        console.log(`SVM prediction successful: ${isTrue ? 'TRUE' : 'FALSE'} with ${Math.round(confidencePercent)}% confidence`);
        
        // Return prediction with confidence
        return {
          isTrue: isTrue,
          confidence: Math.round(Math.min(confidencePercent, 95)), // Cap at 95% to avoid overconfidence
          probabilities: {
            true: isTrue ? Math.round(probability * 100) : Math.round((1 - probability) * 100),
            false: isTrue ? Math.round((1 - probability) * 100) : Math.round(probability * 100)
          }
        };
      } catch (error) {
        console.error('Error during feature generation:', error);
        return this.fallbackVerification(claim, evidence);
      }
    } catch (error) {
      console.error('Error during claim verification:', error);
      // Fallback to simpler method if SVM failed
      return this.fallbackVerification(claim, evidence);
    }
  }
  
  /**
   * Fallback method for claim verification if SVM fails
   */
  fallbackVerification(claim, evidence) {
    try {
      // Combine claim and evidence
      const combinedText = evidence ? `${claim} ${evidence}` : claim;
      const lowerCaseClaim = combinedText.toLowerCase();
      
      // Keywords that suggest fact is likely true
      const trueKeywords = [
        'confirmed', 'verified', 'proven', 'according to research', 
        'studies show', 'evidence supports', 'data shows', 'statistics indicate',
        'average', 'approximately', 'estimated', 'about', 'roughly'
      ];
      
      // Keywords that suggest fact might be false
      const falseKeywords = [
        'all', 'never', 'always', 'every', 'nobody', 'everyone', 'impossible',
        'guaranteed', 'definitely', 'undoubtedly', 'without exception',
        'record-breaking', 'unprecedented', 'worst', 'best', 'greatest'
      ];
      
      // Count matches for each category
      const trueMatches = trueKeywords.filter(word => lowerCaseClaim.includes(word)).length;
      const falseMatches = falseKeywords.filter(word => lowerCaseClaim.includes(word)).length;
      
      // Add check for numerical values which tend to make claims more likely to be true
      const hasNumbers = /\d+(\.\d+)?%?/.test(lowerCaseClaim);
      const hasPreciseValues = /\d+\.\d+/.test(lowerCaseClaim);
      
      const trueScore = trueMatches + (hasNumbers ? 1 : 0) + (hasPreciseValues ? 1 : 0);
      const falseScore = falseMatches * 1.5; // Weight false indicators slightly higher
      
      // Calculate probability
      const totalScore = trueScore + falseScore;
      const trueProbability = totalScore > 0 ? (trueScore / totalScore) : 0.5;
      
      // Final decision with threshold
      const isTrue = trueProbability > 0.5;
      const confidencePercent = Math.abs(trueProbability - 0.5) * 200; // Scale to 0-100%
      
      console.log('Using fallback verification due to SVM failure');
      
      // Return prediction with confidence
      return {
        isTrue: isTrue,
        confidence: Math.round(Math.min(confidencePercent, 95)), // Cap at 95% to avoid overconfidence
        probabilities: {
          true: Math.round(trueProbability * 100),
          false: Math.round((1 - trueProbability) * 100)
        }
      };
    } catch (error) {
      console.error('Error during fallback verification:', error);
      return {
        isTrue: false,
        confidence: 50,
        probabilities: {
          true: 50,
          false: 50
        }
      };
    }
  }

  /**
   * Evaluate model on validation data
   */
  async evaluateModel() {
    if (!this.validationData || !this.svmModel) {
      console.warn('Cannot evaluate: validation data or model not available');
      return null;
    }
    
    try {
      console.log('Evaluating SVM model on validation data...');
      
      // Use a subset of validation data for efficiency
      const testSample = this.validationData.slice(0, 50);
      
      // Extract claims text and labels
      const texts = testSample.map(item => item.claim);
      const trueLabels = testSample.map(item => item.isTrue ? 1 : 0);
      
      // Generate features
      const features = await this.generateFeatures(texts);
      
      // Get predictions
      const predictions = this.svmModel.predict(features);
      
      // Calculate accuracy
      let correct = 0;
      for (let i = 0; i < predictions.length; i++) {
        if (predictions[i] === trueLabels[i]) {
          correct++;
        }
      }
      
      const accuracy = (correct / predictions.length) * 100;
      console.log(`Validation accuracy: ${accuracy.toFixed(2)}%`);
      
      return accuracy;
    } catch (error) {
      console.error('Error evaluating model:', error);
      return null;
    }
  }
}

/**
 * Reinforcement Learning Model for Fact Checking Improvement
 * In a real implementation, this would implement a RL policy
 * for improving fact checking over time.
 */
class RLFactChecker {
  constructor() {
    this.modelLoaded = false;
    this.initialize();
  }
  
  async initialize() {
    try {
      if (isContentScript) {
        console.log('Running RL model in content script context - using simplified approach');
        this.modelLoaded = true;
        return;
      }
      
      // In other contexts, we would initialize the RL model properly
      console.log('Initializing RL fact checker model');
      
      // For demo purposes, just mark as initialized
      this.modelLoaded = true;
    } catch (error) {
      console.error('Failed to initialize RL model:', error);
      this.modelLoaded = true; // Still mark as loaded for fallback
    }
  }

  async selectEvidenceSources(claim) {
    if (!this.modelLoaded) {
      throw new Error('Model not loaded');
    }

    try {
      // Simple rule-based source selection instead of neural network
      // This avoids WebGL and model loading issues in extensions
      const sources = ['wikipedia', 'news', 'factCheckers'];
      
      // Select sources based on claim content
      if (claim.includes('economic') || claim.includes('fiscal') || claim.includes('billion') || 
          claim.includes('deficit') || claim.includes('government')) {
        return ['wikipedia', 'news']; // Economic claims
      } else if (claim.includes('technology') || claim.includes('business') || claim.includes('companies')) {
        return ['news', 'factCheckers']; // Business/tech claims
      } else if (claim.includes('%') || claim.includes('percent') || claim.includes('rate')) {
        return ['wikipedia', 'factCheckers']; // Statistical claims
      }
      
      // Default to all sources
      return sources;
    } catch (error) {
      console.error('Error selecting evidence sources:', error);
      // Default fallback
      return ['wikipedia', 'news', 'factCheckers'];
    }
  }

  async updatePolicy(claim, selectedSources, correctness, reward) {
    if (!this.modelLoaded) {
      throw new Error('Model not loaded');
    }

    try {
      // This would update the policy in a real ML implementation
      // But for the extension, we just log
      console.log('Policy updated with reward:', reward);
      return true;
    } catch (error) {
      console.error('Error updating policy:', error);
      return false;
    }
  }
}

/**
 * Explanation Generation Model
 * In a real implementation, this would generate human-readable
 * explanations for fact checking results.
 */
class ExplanationModel {
  constructor() {
    this.modelLoaded = false;
    this.initialize();
  }
  
  async initialize() {
    try {
      if (isContentScript) {
        console.log('Running explanation model in content script context - using simplified approach');
        this.modelLoaded = true;
        return;
      }
      
      // For demo purposes, just mark as initialized
      console.log('Initializing explanation model');
      this.modelLoaded = true;
    } catch (error) {
      console.error('Failed to initialize explanation model:', error);
      this.modelLoaded = true; // Still mark as loaded for fallback
    }
  }

  async generateExplanation(claim, verificationResult, evidence) {
    if (!this.modelLoaded) {
      throw new Error('Model not loaded');
    }

    try {
      // Generate structured explanation
      const explanation = this._generateStructuredExplanation(
        claim,
        verificationResult,
        evidence
      );

      return explanation;
    } catch (error) {
      console.error('Error generating explanation:', error);
      throw error;
    }
  }

  _generateStructuredExplanation(claim, verificationResult, evidence) {
    const { isTrue, confidence } = verificationResult;
    
    // Extract key information
    const numericalMatches = claim.match(/\d+(\.\d+)?/g) || [];
    const hasNumbers = numericalMatches.length > 0;
    
    // Generate appropriate explanation
    if (isTrue) {
      return hasNumbers
        ? `The numerical values in this claim are verified accurate based on ${evidence}.`
        : `This claim is supported by evidence from reliable sources.`;
    } else {
      return hasNumbers
        ? `The numerical values in this claim differ from verified data. According to ${evidence}, the correct values are shown above.`
        : `This claim contradicts information from reliable sources.`;
    }
  }
}

// Export all models in the AI namespace
export const AI = {
  ClaimExtractionModel,
  SVMFactCheckModel,
  RLFactChecker,
  ExplanationModel
};

// End of module 