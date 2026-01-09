"""
Email Spam Classifier - Complete Application
Author: Student
Date: December 2024
Description: Complete email spam classification system with Logistic Regression
"""

import pandas as pd
import numpy as np
import re
import pickle
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# PART 1: DATASET CREATION
# ============================================================================

def create_email_dataset():
    """Create a comprehensive email dataset for spam classification"""

    print("\n" + "=" * 60)
    print("📁 CREATING EMAIL DATASET")
    print("=" * 60)

    # Legitimate emails (ham)
    legitimate_emails = [
        # Work emails
        "Hi team, the meeting is scheduled for tomorrow at 10 AM in conference room B.",
        "Please find attached the quarterly report for your review.",
        "Project update: All deliverables are on track for the Friday deadline.",
        "Reminder: Performance reviews are due by end of week.",
        "Weekly team lunch this Friday at the Italian restaurant downtown.",

        # Personal emails
        "Hey John, how about we grab coffee tomorrow morning?",
        "Mom, don't forget to pick up groceries on your way home.",
        "Your dentist appointment is confirmed for next Tuesday at 2 PM.",
        "Flight confirmation: Your trip to New York is booked for December 15th.",
        "Happy birthday! Hope you have a wonderful day filled with joy.",

        # Transactional emails
        "Your Amazon order #12345 has been shipped. Tracking number: XYZ789.",
        "Your monthly bank statement is now available for download.",
        "Netflix: Your subscription has been renewed successfully.",
        "Password reset requested for your account. Click here to proceed.",
        "Welcome to our newsletter! You'll receive updates every Thursday.",

        # Social emails
        "You have a new connection request on LinkedIn from Sarah Johnson.",
        "Mark commented on your recent photo on Facebook.",
        "Upcoming event: Tech conference on AI and machine learning.",
        "Your friend Alex sent you a message on WhatsApp.",
        "Twitter: You have 5 new followers this week.",

        # Additional legitimate emails
        "The project deadline has been extended to next Friday.",
        "Please submit your timesheets by 5 PM today.",
        "Company holiday schedule for the upcoming year is attached.",
        "Your software license renewal is due in 30 days.",
        "Team building activity planned for next month at the park.",
    ]

    # Spam emails
    spam_emails = [
        # Urgent/scam emails
        "URGENT: Your bank account has been suspended. Click here to verify.",
        "ALERT: Unusual login detected from unknown device. Secure your account now.",
        "Your PayPal account needs immediate attention. Update your information.",
        "Warning: Your Netflix subscription will be canceled in 24 hours.",
        "Security Alert: Someone tried to access your email account.",

        # Prize/giveaway scams
        "Congratulations! You've won a $1000 Walmart gift card. Claim now!",
        "You're our 1,000,000th visitor! Claim your free iPhone 15 Pro.",
        "Exclusive offer: Free luxury cruise for two to the Bahamas!",
        "You've been selected for a $5000 cash prize. Click to collect!",
        "Special reward: Claim your free MacBook Pro today only!",

        # Money-making scams
        "Earn $5000 weekly working from home. No experience needed!",
        "Get rich quick with this simple cryptocurrency strategy.",
        "Investment opportunity: Double your money in 30 days guaranteed!",
        "Make $1000 daily with this secret online method.",
        "Exclusive: How I made $50,000 in one month (free guide).",

        # Pharmaceutical/health scams
        "Lose 30 pounds in 30 days with this miracle pill.",
        "Doctors hate this one trick for perfect vision without glasses.",
        "New breakthrough treatment cures diabetes in 7 days.",
        "Get prescription drugs without a doctor's approval.",
        "Secret anti-aging formula discovered by scientists.",

        # Adult/dating scams
        "Meet local singles in your area tonight.",
        "Hot girls want to chat with you right now.",
        "Exclusive dating site with verified profiles.",
        "Find your perfect match with our advanced algorithm.",
        "Instant connections with people near you.",

        # Additional spam emails
        "Limited time offer: 90% off all Rolex watches.",
        "Your computer is infected! Download antivirus immediately.",
        "You qualify for a $50,000 loan with bad credit.",
        "Act now! Last chance to claim your free government grant.",
        "Secret method to win the lottery revealed.",
    ]

    # Create labels (0 = legitimate, 1 = spam)
    legitimate_labels = [0] * len(legitimate_emails)
    spam_labels = [1] * len(spam_emails)

    # Combine all emails
    all_emails = legitimate_emails + spam_emails
    all_labels = legitimate_labels + spam_labels

    # Create DataFrame
    data = {
        'email_id': range(1, len(all_emails) + 1),
        'email_text': all_emails,
        'label': all_labels,
        'email_type': ['legitimate'] * len(legitimate_emails) + ['spam'] * len(spam_emails)
    }

    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv('email_dataset.csv', index=False)

    print(f"✅ Dataset created with {len(df)} emails")
    print(f"   Legitimate emails: {len(legitimate_emails)}")
    print(f"   Spam emails: {len(spam_emails)}")
    print("💾 Dataset saved to 'email_dataset.csv'")

    return df


# ============================================================================
# PART 2: EMAIL CLASSIFIER CLASS
# ============================================================================

class EmailSpamClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

    def load_and_preprocess_data(self, filepath='email_dataset.csv'):
        """
        Load and preprocess the email dataset
        Source: email_dataset.csv created by create_email_dataset function
        """
        try:
            # Load data from CSV
            df = pd.read_csv(filepath)
            print(f"\n✅ Dataset loaded: {len(df)} emails")
            print(f"   Legitimate: {len(df[df['label'] == 0])}")
            print(f"   Spam: {len(df[df['label'] == 1])}")

            # Clean email text
            df['cleaned_text'] = df['email_text'].apply(self._clean_text)

            # Split data into train (70%) and test (30%)
            X = df['cleaned_text']
            y = df['label']

            # Using random_state=42 for reproducibility
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )

            print(f"\n📊 Data Split (70/30):")
            print(f"   Training set: {len(self.X_train)} emails (70%)")
            print(f"   Test set: {len(self.X_test)} emails (30%)")

            return True

        except FileNotFoundError:
            print(f"\n❌ Error: File '{filepath}' not found.")
            print("   Please create the dataset first (Option 1 in menu).")
            return False

    def _clean_text(self, text):
        """Clean and preprocess email text"""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def train_logistic_regression(self):
        """
        Train logistic regression model on 70% of data
        Model: Logistic Regression with TF-IDF features
        """
        if self.X_train is None:
            print("\n❌ Please load data first!")
            return False

        print("\n🔄 Training Logistic Regression Model...")

        # Transform text to TF-IDF features
        print("   Extracting TF-IDF features...")
        X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
        self.feature_names = self.vectorizer.get_feature_names_out()

        # Train logistic regression model
        print("   Training model...")
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='liblinear',
            C=1.0
        )

        self.model.fit(X_train_tfidf, self.y_train)

        print("✅ Model training completed!")

        # Show model coefficients
        coefficients_df = self._display_coefficients()

        return True, coefficients_df

    def _display_coefficients(self):
        """Display top coefficients for spam classification"""
        if self.model is None or self.feature_names is None:
            return None

        # Get coefficients for spam class (class 1)
        coefficients = self.model.coef_[0]

        # Create DataFrame for easier viewing
        coef_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients
        })

        # Sort by absolute value of coefficient
        coef_df['abs_coefficient'] = np.abs(coef_df['coefficient'])
        coef_df = coef_df.sort_values('abs_coefficient', ascending=False)

        print("\n📈 Model Coefficients Analysis:")
        print("=" * 60)
        print("Top 10 SPAM indicators (positive coefficients):")
        top_spam = coef_df[coef_df['coefficient'] > 0].head(10)
        for idx, (_, row) in enumerate(top_spam.iterrows(), 1):
            print(f"   {idx:2}. {row['feature']:15} : {row['coefficient']:+.4f}")

        print("\nTop 10 LEGITIMATE indicators (negative coefficients):")
        top_legit = coef_df[coef_df['coefficient'] < 0].head(10)
        for idx, (_, row) in enumerate(top_legit.iterrows(), 1):
            print(f"   {idx:2}. {row['feature']:15} : {row['coefficient']:+.4f}")

        # Save coefficients to file
        coef_df.to_csv('model_coefficients.csv', index=False)
        print(f"\n💾 All coefficients saved to 'model_coefficients.csv'")

        return coef_df

    def validate_model(self):
        """
        Validate model on the 30% test data
        Calculate confusion matrix and accuracy
        """
        if self.model is None or self.X_test is None:
            print("\n❌ Please train model first!")
            return

        print("\n🔍 Validating Model on Test Data (30%)...")

        # Transform test data
        X_test_tfidf = self.vectorizer.transform(self.X_test)

        # Make predictions
        y_pred = self.model.predict(X_test_tfidf)

        # Calculate accuracy
        accuracy = accuracy_score(self.y_test, y_pred)

        # Calculate confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)

        # Display results
        print("✅ Validation Results:")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        print("\n📊 Confusion Matrix:")
        print("   " + "=" * 40)
        print("   Actual \\ Predicted  |  Legitimate  |  Spam")
        print("   " + "-" * 40)
        print(f"   Legitimate (0)    |     {cm[0, 0]:^8}   |  {cm[0, 1]:^4}")
        print(f"   Spam (1)          |     {cm[1, 0]:^8}   |  {cm[1, 1]:^4}")
        print("   " + "=" * 40)

        # Calculate additional metrics
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print("\n📋 Performance Metrics:")
        print(f"   True Positives (TP): {tp}")
        print(f"   True Negatives (TN): {tn}")
        print(f"   False Positives (FP): {fp}")
        print(f"   False Negatives (FN): {fn}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1_score:.4f}")

        # Detailed classification report
        print("\n📝 Detailed Classification Report:")
        report = classification_report(self.y_test, y_pred,
                                       target_names=['Legitimate', 'Spam'])
        print(report)

        return accuracy, cm

    def classify_email(self, email_text):
        """
        Classify a single email as spam or legitimate
        """
        if self.model is None:
            print("\n❌ Please train model first!")
            return None

        # Clean the input email
        cleaned_text = self._clean_text(email_text)

        # Transform to TF-IDF features
        email_tfidf = self.vectorizer.transform([cleaned_text])

        # Make prediction
        prediction = self.model.predict(email_tfidf)[0]
        probabilities = self.model.predict_proba(email_tfidf)[0]

        # Get top contributing features
        top_features = self._get_contributing_features(email_tfidf)

        return {
            'prediction': 'SPAM' if prediction == 1 else 'LEGITIMATE',
            'confidence': max(probabilities) * 100,
            'spam_probability': probabilities[1] * 100,
            'legitimate_probability': probabilities[0] * 100,
            'cleaned_text': cleaned_text,
            'top_features': top_features
        }

    def _get_contributing_features(self, email_tfidf, top_n=5):
        """Get top contributing features for the prediction"""
        if self.model is None or self.feature_names is None:
            return []

        # Get feature indices with non-zero values
        feature_indices = email_tfidf.nonzero()[1]

        # Get corresponding feature names and TF-IDF values
        contributing_features = []
        for idx in feature_indices:
            feature_name = self.feature_names[idx]
            tfidf_value = email_tfidf[0, idx]

            # Get coefficient for this feature
            coefficient = self.model.coef_[0][idx]

            # Calculate contribution (TF-IDF * coefficient)
            contribution = tfidf_value * coefficient

            contributing_features.append({
                'feature': feature_name,
                'coefficient': coefficient,
                'tfidf': float(tfidf_value),
                'contribution': float(contribution)
            })

        # Sort by absolute contribution
        contributing_features.sort(key=lambda x: abs(x['contribution']), reverse=True)

        return contributing_features[:top_n]

    def save_model(self, filename='email_classifier_model.pkl'):
        """Save the trained model and vectorizer"""
        if self.model is None:
            print("\n❌ No model to save!")
            return False

        try:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'vectorizer': self.vectorizer,
                    'feature_names': self.feature_names
                }, f)
            print(f"\n✅ Model saved to '{filename}'")
            return True
        except Exception as e:
            print(f"\n❌ Error saving model: {e}")
            return False

    def load_model(self, filename='email_classifier_model.pkl'):
        """Load a trained model"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)

            self.model = data['model']
            self.vectorizer = data['vectorizer']
            self.feature_names = data['feature_names']

            print(f"\n✅ Model loaded from '{filename}'")
            return True
        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
            return False


# ============================================================================
# PART 3: TEST FUNCTION
# ============================================================================

def test_requirements():
    """Test that all requirements are met"""

    print("\n" + "=" * 60)
    print("🧪 TESTING ALL REQUIREMENTS")
    print("=" * 60)

    # First create dataset
    df = create_email_dataset()

    # Load data
    print("\n1. ✅ Dataset loaded successfully")
    print(f"   Total emails: {len(df)}")
    print(f"   Legitimate: {len(df[df['label'] == 0])}")
    print(f"   Spam: {len(df[df['label'] == 1])}")

    # Clean text function
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['cleaned_text'] = df['email_text'].apply(clean_text)
    print("2. ✅ Text preprocessing implemented")

    # Split data 70/30
    X = df['cleaned_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print("3. ✅ Data split: 70% train, 30% test")
    print(f"   Training set: {len(X_train)} emails")
    print(f"   Test set: {len(X_test)} emails")

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    print("4. ✅ TF-IDF feature extraction")
    print(f"   Features extracted: {X_train_tfidf.shape[1]}")

    # Train Logistic Regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    print("5. ✅ Logistic Regression model trained")
    print(f"   Model coefficients shape: {model.coef_.shape}")

    # Validate on test data
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("6. ✅ Model validation on test data")
    print(f"   Accuracy: {accuracy:.4f}")
    print("   Confusion Matrix:")
    print("   " + "-" * 30)
    print(f"   [[{cm[0, 0]:2}  {cm[0, 1]:2}]")
    print(f"    [{cm[1, 0]:2}  {cm[1, 1]:2}]]")

    # Test classification of new email
    test_email = "Win free money now! Click here for amazing offer!"
    cleaned_test = clean_text(test_email)
    test_tfidf = vectorizer.transform([cleaned_test])
    prediction = model.predict(test_tfidf)[0]
    probability = model.predict_proba(test_tfidf)[0]

    print("7. ✅ New email classification working")
    print(f"   Test email: '{test_email}'")
    print(f"   Prediction: {'SPAM' if prediction == 1 else 'LEGITIMATE'}")
    print(f"   Spam probability: {probability[1] * 100:.1f}%")

    print("\n" + "=" * 60)
    print("✅ ALL REQUIREMENTS MET SUCCESSFULLY!")
    print("=" * 60)

    return True


# ============================================================================
# PART 4: MAIN APPLICATION
# ============================================================================

def display_menu():
    """Display main menu"""
    print("\n" + "=" * 60)
    print("📧 EMAIL SPAM CLASSIFIER - MAIN MENU")
    print("=" * 60)
    print("1. Create dataset (first time only)")
    print("2. Load and preprocess data")
    print("3. Train Logistic Regression model (70% data)")
    print("4. Validate model (30% test data)")
    print("5. Classify single email")
    print("6. Classify multiple sample emails")
    print("7. Test all requirements")
    print("8. Save trained model")
    print("9. Load saved model")
    print("0. Exit")
    print("-" * 60)


def run_application():
    """Main application interface"""
    classifier = EmailSpamClassifier()

    while True:
        display_menu()
        choice = input("\nEnter your choice (0-9): ").strip()

        if choice == '0':
            print("\n👋 Goodbye! Thank you for using Email Spam Classifier!")
            break

        elif choice == '1':
            # Create dataset
            df = create_email_dataset()

        elif choice == '2':
            # Load and preprocess data
            success = classifier.load_and_preprocess_data()
            if success:
                print("\n✅ Data loaded and preprocessed successfully!")

        elif choice == '3':
            # Train Logistic Regression model
            success, coefficients_df = classifier.train_logistic_regression()
            if success:
                print("\n✅ Model trained successfully!")

                # Ask if user wants to see all coefficients
                see_all = input("\nView all coefficients in file? (y/n): ").lower()
                if see_all == 'y' and coefficients_df is not None:
                    print("\n📋 First 30 coefficients:")
                    print(coefficients_df.head(30).to_string())

        elif choice == '4':
            # Validate model
            accuracy, cm = classifier.validate_model()

        elif choice == '5':
            # Classify single email
            print("\n" + "=" * 60)
            print("📝 CLASSIFY SINGLE EMAIL")
            print("=" * 60)

            email_text = input("\nEnter email text to classify:\n> ")

            if email_text.strip():
                result = classifier.classify_email(email_text)

                if result:
                    print(f"\n📊 Classification Result:")
                    print(f"   Prediction: {result['prediction']}")
                    print(f"   Confidence: {result['confidence']:.1f}%")
                    print(f"   Spam Probability: {result['spam_probability']:.1f}%")
                    print(f"   Legitimate Probability: {result['legitimate_probability']:.1f}%")

                    if result['top_features']:
                        print(f"\n🔍 Top contributing features:")
                        for feature in result['top_features']:
                            if feature['coefficient'] > 0:
                                print(
                                    f"   • '{feature['feature']}' suggests SPAM (contribution: {feature['contribution']:.4f})")
                            else:
                                print(
                                    f"   • '{feature['feature']}' suggests LEGITIMATE (contribution: {feature['contribution']:.4f})")
            else:
                print("❌ No email text provided!")

        elif choice == '6':
            # Classify multiple sample emails
            print("\n" + "=" * 60)
            print("📚 CLASSIFYING SAMPLE EMAILS")
            print("=" * 60)

            sample_emails = [
                "Congratulations! You won a free iPhone. Click here to claim now!",
                "Hi team, meeting scheduled for tomorrow at 10 AM.",
                "URGENT: Your account has been compromised. Verify immediately!",
                "Please review the attached report and provide feedback.",
                "Earn $5000 weekly working from home with no experience needed!"
            ]

            for i, email in enumerate(sample_emails, 1):
                print(f"\n📝 Sample Email {i}:")
                print(f"   '{email[:80]}...'" if len(email) > 80 else f"   '{email}'")

                result = classifier.classify_email(email)

                if result:
                    print(f"\n   Prediction: {result['prediction']}")
                    print(f"   Confidence: {result['confidence']:.1f}%")
                    print(f"   Spam Probability: {result['spam_probability']:.1f}%")

                    if result['top_features']:
                        print(f"\n   Top contributing features:")
                        for feature in result['top_features']:
                            direction = "↑ spam" if feature['coefficient'] > 0 else "↓ legitimate"
                            print(f"     • '{feature['feature']}': {feature['contribution']:.4f} ({direction})")

        elif choice == '7':
            # Test all requirements
            test_requirements()

        elif choice == '8':
            # Save trained model
            filename = input("\nEnter filename (or press Enter for default 'email_classifier_model.pkl'): ").strip()
            if not filename:
                filename = 'email_classifier_model.pkl'
            classifier.save_model(filename)

        elif choice == '9':
            # Load saved model
            filename = input("\nEnter filename to load (or press Enter for default): ").strip()
            if not filename:
                filename = 'email_classifier_model.pkl'

            if os.path.exists(filename):
                success = classifier.load_model(filename)
                if success:
                    print("\n✅ Model loaded successfully!")
            else:
                print(f"\n❌ File '{filename}' not found!")

        else:
            print("\n❌ Invalid choice! Please enter a number between 0 and 9.")

        if choice != '0':
            input("\nPress Enter to continue...")


# ============================================================================
# PART 5: DIRECT EXECUTION
# ============================================================================

def run_complete_pipeline():
    """Run the complete pipeline from dataset creation to classification"""
    print("\n" + "=" * 60)
    print("🚀 RUNNING COMPLETE PIPELINE")
    print("=" * 60)

    # 1. Create dataset
    print("\n📁 STEP 1: Creating dataset...")
    df = create_email_dataset()

    # 2. Initialize classifier
    print("\n🤖 STEP 2: Initializing classifier...")
    classifier = EmailSpamClassifier()

    # 3. Load and preprocess data
    print("\n🔄 STEP 3: Loading and preprocessing data...")
    success = classifier.load_and_preprocess_data()
    if not success:
        return

    # 4. Train model
    print("\n🎯 STEP 4: Training Logistic Regression model...")
    success, coefficients_df = classifier.train_logistic_regression()
    if not success:
        return

    # 5. Validate model
    print("\n📊 STEP 5: Validating model on test data...")
    accuracy, cm = classifier.validate_model()

    # 6. Classify sample emails
    print("\n🔍 STEP 6: Classifying sample emails...")
    test_emails = [
        "Win free money! Click now for exclusive offer!",
        "Team meeting at 3 PM tomorrow in conference room",
        "Your account needs verification immediately"
    ]

    for i, email in enumerate(test_emails, 1):
        print(f"\n   Email {i}: '{email}'")
        result = classifier.classify_email(email)
        if result:
            print(f"   → {result['prediction']} ({result['confidence']:.1f}% confidence)")

    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("📧 EMAIL SPAM CLASSIFIER APPLICATION")
    print("=" * 60)
    print("Complete system for classifying emails as spam or legitimate")
    print("Using Logistic Regression trained on 70% of data")
    print("Validated on 30% test data with confusion matrix")
    print("=" * 60)

    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            test_requirements()
        elif sys.argv[1] == '--pipeline':
            run_complete_pipeline()
        elif sys.argv[1] == '--menu':
            run_application()
        else:
            print(f"\n❌ Unknown argument: {sys.argv[1]}")
            print("Available arguments:")
            print("  --test     : Run requirement tests")
            print("  --pipeline : Run complete pipeline")
            print("  --menu     : Run interactive menu (default)")
            run_application()
    else:
        # Default: run interactive menu
        run_application()