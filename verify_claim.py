#!/usr/bin/env python
import sys
import time
from claim_verifier import ClaimVerifier

def main():
    print("Loading optimized kernel SVM claim verification model...")
    verifier = ClaimVerifier()
    
    try:
        load_start = time.time()
        verifier.load_model('claim_verifier_model.joblib')
        load_time = time.time() - load_start
        print(f"Model loaded successfully in {load_time:.3f} seconds!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please make sure you've run run_liar_verifier.py first to train the model.")
        sys.exit(1)
    
    print("\nOptimized Kernel SVM Claim Verification Tool")
    print("==========================================")
    print("This tool uses a state-of-the-art SVM model with")
    print("optimized kernels, kernel approximation techniques,")
    print("and advanced text features to classify")
    print("political statements as true or false.")
    print("\nEnter a claim to verify, or type 'exit' to quit.")
    
    # Keep track of prediction times for performance monitoring
    prediction_times = []
    
    while True:
        print("\nEnter your claim:")
        claim = input("> ")
        
        if claim.lower() in ['exit', 'quit', 'q']:
            if prediction_times:
                avg_time = sum(prediction_times) / len(prediction_times)
                print(f"\nPerformance stats: Average prediction time: {avg_time*1000:.2f} ms")
                print(f"Fastest: {min(prediction_times)*1000:.2f} ms, Slowest: {max(prediction_times)*1000:.2f} ms")
            print("Exiting Claim Verification Tool. Goodbye!")
            break
            
        if not claim.strip():
            print("Please enter a valid claim.")
            continue
        
        try:
            predict_start = time.time()
            verdict, confidence = verifier.predict_claim(claim)
            predict_time = time.time() - predict_start
            prediction_times.append(predict_time)
            
            print("\nRESULTS:")
            print(f"Claim: {claim}")
            print(f"Verdict: {verdict.upper()}")
            print(f"Confidence: {confidence:.2%}")
            print(f"Prediction time: {predict_time*1000:.2f} ms")
            
            # Provide interpretation based on confidence
            if confidence > 0.8:
                confidence_level = "very high"
            elif confidence > 0.7:
                confidence_level = "high"
            elif confidence > 0.6:
                confidence_level = "moderate"
            else:
                confidence_level = "low"
                
            print(f"The model has {confidence_level} confidence in this prediction.")
            
            if confidence < 0.55:
                print("Note: This prediction has low confidence and may not be reliable.")
                
            # Provide explanation based on basic linguistic features
            word_count = len(claim.split())
            question_count = claim.count('?')
            exclamation_count = claim.count('!')
            capital_ratio = sum(1 for c in claim if c.isupper()) / len(claim) if len(claim) > 0 else 0
            
            print("\nClaim analysis:")
            if word_count < 5:
                print("- Very short claims often lack context for accurate verification")
            elif word_count > 50:
                print("- Very long claims may contain multiple statements with varying truthfulness")
                
            if question_count > 0:
                print("- Questions often aren't verifiable factual claims")
                
            if exclamation_count > 0:
                print("- Exclamatory statements may indicate emotional content rather than factual information")
                
            if capital_ratio > 0.5:
                print("- Heavy use of capital letters can indicate hyperbole or exaggeration")
            
            # Performance suggestion based on prediction time
            if predict_time > 0.1:  # If prediction takes more than 100ms
                print("\nPerformance note: This claim took longer than usual to process.")
                
        except Exception as e:
            print(f"Error processing claim: {e}")

if __name__ == "__main__":
    main() 