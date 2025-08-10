#!/usr/bin/env python3
"""
EdgeShrink AI - Quality Control Demo

Automated visual inspection system for manufacturing quality control 
using edge AI for real-time defect detection and assessment.

Market: Manufacturing automation
"""

import argparse
import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.vision_language_engine import load_model
except ImportError:
    print("Warning: EdgeShrink AI models not available. Running in demo mode.")
    load_model = None

class QualityInspector:
    """Industrial Quality Control Inspector using EdgeShrink AI"""
    
    def __init__(self):
        """Initialize the quality inspector with multimodal QA model"""
        self.model = None
        if load_model:
            try:
                print("Loading EdgeShrink AI Multimodal QA Model...")
                self.model = load_model("models/multimodal_qa_model")
                print("Quality control model loaded successfully!")
            except Exception as e:
                print(f"Could not load model: {e}")
                print("Running in demo mode with simulated inspection.")
    
    def inspect_product(self, image_path, quality_questions=None):
        """
        Automated quality inspection using visual QA
        
        Args:
            image_path: Path to product image
            quality_questions: List of inspection questions
            
        Returns:
            dict: Comprehensive quality assessment report
        """
        if not os.path.exists(image_path):
            return {
                'error': f"Product image not found: {image_path}",
                'suggestion': "Please provide a valid image path"
            }
        
        if quality_questions is None:
            quality_questions = self.get_standard_quality_questions()
        
        print(f"Inspecting product: {image_path}")
        print(f"Running {len(quality_questions)} quality checks...")
        
        inspection_results = {}
        
        if self.model:
            try:
                # Use actual EdgeShrink AI model for inspection
                for question in quality_questions:
                    answer = self.model.answer_question(image_path, question)
                    inspection_results[question] = answer
            except Exception as e:
                print(f"Model inference error: {e}")
                return self.generate_demo_inspection(image_path, quality_questions)
        else:
            # Demo mode with simulated inspection
            return self.generate_demo_inspection(image_path, quality_questions)
        
        return self.generate_quality_report(inspection_results, image_path)
    
    def get_standard_quality_questions(self):
        """Standard quality control questions for manufacturing"""
        return [
            "Are there any visible defects in this component?",
            "Is the surface finish acceptable and smooth?",
            "Are all required markings and labels present?",
            "Is the component properly aligned and positioned?",
            "Are there any scratches, dents, or damage visible?",
            "Does the color and appearance match specifications?",
            "Are all edges clean and properly finished?",
            "Is the overall build quality acceptable?"
        ]
    
    def generate_quality_report(self, results, image_path):
        """Generate comprehensive quality assessment report"""
        defects = []
        quality_score = 0
        total_checks = len(results)
        
        for question, answer in results.items():
            if any(keyword in answer.lower() for keyword in ['defect', 'damaged', 'poor', 'unacceptable', 'missing']):
                defects.append({
                    'check': question,
                    'finding': answer,
                    'severity': self.assess_severity(answer)
                })
            else:
                quality_score += 1
        
        overall_score = quality_score / total_checks if total_checks > 0 else 0
        pass_fail = overall_score >= 0.8  # 80% pass threshold
        
        return {
            'product_image': image_path,
            'inspection_timestamp': datetime.now().isoformat(),
            'overall_quality_score': round(overall_score * 100, 1),
            'pass_fail_status': 'PASS' if pass_fail else 'FAIL',
            'total_checks_performed': total_checks,
            'defects_found': len(defects),
            'defect_details': defects,
            'quality_assessment': self.generate_assessment_summary(overall_score, defects),
            'recommendations': self.generate_recommendations(defects, overall_score),
            'market_value': "Industrial QC system - $1M+ installation potential"
        }
    
    def assess_severity(self, finding):
        """Assess severity of quality issue"""
        finding_lower = finding.lower()
        if any(word in finding_lower for word in ['critical', 'major', 'severe', 'damaged']):
            return 'HIGH'
        elif any(word in finding_lower for word in ['minor', 'slight', 'small']):
            return 'LOW'
        else:
            return 'MEDIUM'
    
    def generate_assessment_summary(self, score, defects):
        """Generate human-readable assessment summary"""
        if score >= 0.9:
            return "Excellent quality - meets all specifications"
        elif score >= 0.8:
            return "Good quality - minor issues detected"
        elif score >= 0.6:
            return "Acceptable quality - some defects require attention"
        else:
            return "Poor quality - significant issues detected"
    
    def generate_recommendations(self, defects, score):
        """Generate actionable recommendations"""
        recommendations = []
        
        if score >= 0.8:
            recommendations.append("Component approved for shipment")
            if defects:
                recommendations.append("Monitor minor issues in future production")
        else:
            recommendations.append("Component requires rework before approval")
            
            high_severity = [d for d in defects if d['severity'] == 'HIGH']
            if high_severity:
                recommendations.append("Address critical defects immediately")
            
            recommendations.append("Review manufacturing process for quality improvements")
        
        return recommendations
    
    def generate_demo_inspection(self, image_path, quality_questions):
        """Generate demo inspection results for showcase"""
        filename = os.path.basename(image_path).lower()
        
        # Simulate different quality scenarios based on filename
        if 'good' in filename or 'pass' in filename:
            # High quality component
            results = {}
            for question in quality_questions:
                if 'defect' in question:
                    results[question] = "No visible defects detected"
                elif 'surface' in question:
                    results[question] = "Surface finish is smooth and acceptable"
                elif 'marking' in question:
                    results[question] = "All required markings are clearly present"
                elif 'alignment' in question:
                    results[question] = "Component is properly aligned"
                elif 'scratch' in question:
                    results[question] = "No scratches or damage visible"
                elif 'color' in question:
                    results[question] = "Color matches specifications perfectly"
                elif 'edges' in question:
                    results[question] = "All edges are clean and properly finished"
                else:
                    results[question] = "Quality meets specifications"
            
        elif 'defect' in filename or 'fail' in filename:
            # Poor quality component
            results = {}
            for question in quality_questions:
                if 'defect' in question:
                    results[question] = "Multiple surface defects detected"
                elif 'surface' in question:
                    results[question] = "Surface finish is rough and unacceptable"
                elif 'marking' in question:
                    results[question] = "Required markings are missing or unclear"
                elif 'scratch' in question:
                    results[question] = "Several scratches and dents visible"
                else:
                    results[question] = "Does not meet quality specifications"
                    
        else:
            # Average quality component
            results = {}
            defect_questions = quality_questions[:2]  # First 2 questions show issues
            
            for i, question in enumerate(quality_questions):
                if i < 2:  # Some issues
                    if 'defect' in question:
                        results[question] = "Minor surface defect detected in corner"
                    elif 'surface' in question:
                        results[question] = "Surface finish acceptable with minor roughness"
                    else:
                        results[question] = "Minor issue detected but within tolerance"
                else:  # Rest are good
                    results[question] = "Meets quality specifications"
        
        return self.generate_quality_report(results, image_path)

def create_sample_images():
    """Create sample image directories and placeholder files"""
    sample_dir = Path("factory_images")
    sample_dir.mkdir(exist_ok=True)
    
    sample_files = [
        "component_good.jpg",
        "component_defect.jpg",
        "pcb_inspection.jpg",
        "automotive_part.jpg",
        "component.jpg"
    ]
    
    for filename in sample_files:
        sample_path = sample_dir / filename
        if not sample_path.exists():
            # Create placeholder file
            sample_path.write_text(f"# Placeholder for {filename}\n# Replace with actual component image")
            print(f"Created placeholder: {sample_path}")

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="EdgeShrink AI Quality Control Demo")
    parser.add_argument("--image", required=True, help="Path to component image")
    parser.add_argument("--questions", help="JSON file with custom quality questions")
    parser.add_argument("--create-samples", action="store_true", help="Create sample image placeholders")
    parser.add_argument("--output", help="Save report to JSON file")
    
    args = parser.parse_args()
    
    if args.create_samples:
        create_sample_images()
        print("Sample image placeholders created in factory_images/")
        return
    
    # Load custom questions if provided
    quality_questions = None
    if args.questions:
        try:
            with open(args.questions, 'r') as f:
                data = json.load(f)
                quality_questions = data.get('questions', None)
        except Exception as e:
            print(f"Error loading questions file: {e}")
            print("Using standard quality questions.")
    
    # Initialize quality inspector
    inspector = QualityInspector()
    
    # Perform quality inspection
    print("EDGESHRINK AI - INDUSTRIAL QUALITY CONTROL DEMO")
    
    report = inspector.inspect_product(args.image, quality_questions)
    
    if 'error' in report:
        print(f"Error: {report['error']}")
        print(f"Suggestion: {report['suggestion']}")
        print("\nTip: Use --create-samples to generate placeholder images")
        return
    
    # Display inspection report
    print(f"\nQuality Inspection Report")
    print(f"Product: {report['product_image']}")
    print(f"Timestamp: {report['inspection_timestamp']}")
    print(f"Overall Quality Score: {report['overall_quality_score']}%")
    print(f"Status: {report['pass_fail_status']}")
    print(f"Total Checks: {report['total_checks_performed']}")
    print(f"Defects Found: {report['defects_found']}")
    
    print(f"\nQuality Assessment: {report['quality_assessment']}")
    
    if report['defect_details']:
        print(f"\nDefect Details:")
        for i, defect in enumerate(report['defect_details'], 1):
            print(f"  {i}. {defect['check']}")
            print(f"     Finding: {defect['finding']}")
            print(f"     Severity: {defect['severity']}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print(f"\nMarket Application: {report['market_value']}")
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.output}")
    
if __name__ == "__main__":
    main()
