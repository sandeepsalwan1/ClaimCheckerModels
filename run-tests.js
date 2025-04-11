/**
 * Auto-run all tests and commands
 * 
 * This script automatically runs all tests for the NewsFactChecker project.
 */

import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

// Tests to run
const tests = [
  {
    name: 'SVM Model Test',
    command: 'node test-svm.js'
  }
];

// Run all tests sequentially
async function runAllTests() {
  console.log('Starting automated tests...\n');
  
  let allTestsPassed = true;
  
  for (const test of tests) {
    console.log(`Running ${test.name}...`);
    
    try {
      const { stdout, stderr } = await execAsync(test.command);
      
      if (stdout) console.log(stdout);
      if (stderr) console.error(stderr);
      
      console.log(`✅ ${test.name} completed successfully\n`);
    } catch (error) {
      console.error(`❌ ${test.name} failed`);
      if (error.stdout) console.log(error.stdout);
      if (error.stderr) console.error(error.stderr);
      console.log('\n');
      
      allTestsPassed = false;
    }
  }
  
  if (allTestsPassed) {
    console.log('All tests passed successfully! 🎉');
    process.exit(0);
  } else {
    console.error('Some tests failed. Please check the output above.');
    process.exit(1);
  }
}

// Run the tests
runAllTests(); 