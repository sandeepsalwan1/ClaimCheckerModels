/**
 * LIAR Dataset Processing
 * 
 * This module handles loading and processing the LIAR dataset for fact verification.
 * The LIAR dataset contains human-labeled short statements from PolitiFact.com's 
 * API, and each statement is evaluated by a professional fact-checker for its truthfulness.
 * 
 * For testing purposes, we're using a small local dataset.
 */

// For loading and parsing CSV data
import papaparse from 'papaparse';
const Papa = papaparse;

/**
 * Loads a local LIAR dataset (for testing purposes)
 * @returns {Promise<Array>} The loaded dataset
 */
async function loadLiarDataset() {
  try {
    console.log('Loading local LIAR dataset...');
    
    // Local dataset for testing
    const localData = [
      { id: "1", label: "true", statement: "The unemployment rate is at a 50-year low.", subject: "economy", speaker: "politician", context: "speech" },
      { id: "2", label: "true", statement: "The federal debt has increased by over $5 trillion in the last 4 years.", subject: "economy", speaker: "journalist", context: "article" },
      { id: "3", label: "mostly-true", statement: "Renewable energy now accounts for about 20% of electricity generated in the US.", subject: "energy", speaker: "scientist", context: "interview" },
      { id: "4", label: "mostly-true", statement: "The average college graduate earns over $1 million more in their lifetime than non-graduates.", subject: "education", speaker: "educator", context: "speech" },
      { id: "5", label: "half-true", statement: "The top 1% of earners pay 40% of all income taxes.", subject: "taxes", speaker: "politician", context: "debate" },
      { id: "6", label: "half-true", statement: "Healthcare costs have risen at twice the rate of inflation for the past decade.", subject: "healthcare", speaker: "industry", context: "report" },
      { id: "7", label: "barely-true", statement: "The average family spends over $10,000 per year on healthcare costs.", subject: "healthcare", speaker: "politician", context: "speech" },
      { id: "8", label: "barely-true", statement: "Violent crime has increased by 50% in major cities over the last 5 years.", subject: "crime", speaker: "politician", context: "interview" },
      { id: "9", label: "false", statement: "Tax cuts always lead to increased government revenue.", subject: "taxes", speaker: "politician", context: "debate" },
      { id: "10", label: "false", statement: "Over 80% of the federal budget goes to foreign aid.", subject: "budget", speaker: "politician", context: "rally" },
      { id: "11", label: "pants-fire", statement: "The economy had negative growth for 10 consecutive quarters.", subject: "economy", speaker: "politician", context: "debate" },
      { id: "12", label: "pants-fire", statement: "Unemployment is currently at 30% nationwide.", subject: "economy", speaker: "politician", context: "social media" },
      { id: "13", label: "true", statement: "The human brain has over 80 billion neurons.", subject: "science", speaker: "scientist", context: "interview" },
      { id: "14", label: "false", statement: "The average smartphone user checks their phone over 1,000 times per day.", subject: "technology", speaker: "journalist", context: "article" },
      { id: "15", label: "true", statement: "Global carbon emissions reached a record high in 2023.", subject: "climate", speaker: "scientist", context: "report" },
      { id: "16", label: "false", statement: "Electric vehicles produce more pollution than gas vehicles when manufacturing is considered.", subject: "environment", speaker: "industry", context: "interview" },
      { id: "17", label: "mostly-true", statement: "The average American spends over 5 hours per day on digital devices.", subject: "technology", speaker: "researcher", context: "report" },
      { id: "18", label: "half-true", statement: "The US federal minimum wage has lost 30% of its purchasing power since 2000.", subject: "economy", speaker: "politician", context: "speech" },
      { id: "19", label: "barely-true", statement: "Over half of all startups fail within the first year.", subject: "business", speaker: "journalist", context: "article" },
      { id: "20", label: "pants-fire", statement: "The average temperature has not increased globally in the past 50 years.", subject: "climate", speaker: "politician", context: "speech" }
    ];
    
    console.log(`Successfully loaded ${localData.length} fact-check records`);
    return localData;
  } catch (error) {
    console.error('Error loading local LIAR dataset:', error);
    throw new Error('Failed to load LIAR dataset. The application requires this dataset to function properly.');
  }
}

/**
 * Processes and prepares the LIAR dataset for model training
 * @param {Array} rawData - The raw dataset records
 * @returns {Array} Processed records with binary labels
 */
function processLiarDataset(rawData) {
  if (!rawData || !Array.isArray(rawData) || rawData.length === 0) {
    throw new Error('Invalid dataset provided');
  }
  
  // Map the 6-class labels to binary labels (true/false)
  // true, mostly-true → true (1)
  // half-true, barely-true, false, pants-fire → false (0)
  return rawData.map(record => {
    // Get the primary text fields containing the claim
    const statement = record.statement || '';
    
    // Map the 6-class label to binary
    let isTrue = false;
    const label = record.label ? record.label.toLowerCase() : '';
    
    if (label === 'true' || label === 'mostly-true') {
      isTrue = true;
    }
    
    return {
      id: record.id,
      claim: statement,
      isTrue: isTrue,
      originalLabel: label,
      subject: record.subject || '',
      speaker: record.speaker || '',
      context: record.context || ''
    };
  }).filter(record => record.claim && record.claim.length > 10);
}

/**
 * Splits dataset into training and validation sets
 * @param {Array} dataset - The full processed dataset
 * @param {number} validationRatio - Ratio of validation set (0-1)
 * @returns {Object} Object with training and validation arrays
 */
function splitDataset(dataset, validationRatio = 0.2) {
  // Shuffle the dataset
  const shuffled = [...dataset].sort(() => 0.5 - Math.random());
  
  // Calculate split index
  const splitIndex = Math.floor(shuffled.length * (1 - validationRatio));
  
  // Split into training and validation sets
  const training = shuffled.slice(0, splitIndex);
  const validation = shuffled.slice(splitIndex);
  
  // Log data distribution stats
  const trainingTrueCount = training.filter(item => item.isTrue).length;
  const validationTrueCount = validation.filter(item => item.isTrue).length;
  
  console.log('Dataset split statistics:');
  console.log(`Training set: ${training.length} records (${trainingTrueCount} true, ${training.length - trainingTrueCount} false)`);
  console.log(`Validation set: ${validation.length} records (${validationTrueCount} true, ${validation.length - validationTrueCount} false)`);
  
  return { training, validation };
}

export { loadLiarDataset, processLiarDataset, splitDataset }; 