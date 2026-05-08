# main.py
import os
import argparse
from src.adapters.preprocessing.text_cleaner import NLTKPreprocessor
from src.adapters.feature_extraction.tfidf_extractor import TfidfExtractor
from src.adapters.classification.sklearn_model import LogisticRegressionClassifier
from src.adapters.persistence.file_storage import JoblibModelRepository
from src.use_cases.phishing_analyzer import PhishingAnalyzer
from src.adapters.web.flask_app import create_app
from src.core.entities import LABEL_SAFE, LABEL_PHISHING, LABEL_PROMOTIONAL

def train_dummy_model():
    print("Initializing mapping and training dummy model...")
    texts = [
        # --- PHISHING EXAMPLES ---
        # Prize / Lottery scams
        "Win a free iPhone now! Click here to claim your prize.",
        "Congratulations! You have been selected to receive a $1000 Amazon gift card. Click here.",
        "You have won the international lottery! Reply with your bank details to transfer the funds.",
        "You are today's lucky winner. Claim your free iPad by clicking this link immediately.",
        "WINNER ANNOUNCEMENT: Your email has won $500,000. Contact us to claim your winnings.",
        "Exclusive offer: You've won a free vacation package. Click to verify your identity and claim.",

        # Account / Bank / Credential phishing
        "Your bank account has been compromised. Verify immediately.",
        "URGENT: Your PayPal account has been limited. Please confirm your details to restore access.",
        "Security Alert: Unauthorized login attempt on your Apple ID. Reset your password now.",
        "Action Required: Unusual activity detected on your Chase account. Click to secure your account.",
        "Your Netflix subscription has been suspended. Update your payment details to continue streaming.",
        "Your Amazon account will be locked. Verify your information at the link below to prevent this.",
        "We noticed a sign-in attempt from an unknown device. Click here to confirm it was you.",
        "Your Microsoft account password needs to be verified. Login immediately to avoid suspension.",
        "Dear customer, your credit card has been charged $890. Dispute this charge at the link below.",

        # Package / Delivery scams
        "You have a pending package delivery. Pay the $2.99 shipping fee at this link to receive it.",
        "USPS: Your parcel could not be delivered. Click the link to reschedule.",
        "DHL Notification: Your package is held at customs. Pay the duty fee to release it.",
        "FedEx: Action required to release your shipment. Click here to confirm your delivery address.",

        # Tax / Refund scams
        "Important notification regarding your tax refund. Fill out the attached form to process.",
        "IRS Notice: You are eligible for a tax refund of $320. Click here to claim it.",
        "HMRC Tax Refund: You are owed £450. Click the secure link below to submit your claim.",

        # Job / Financial scams
        "Work from home and earn $5000 per week. No experience needed. Apply now.",
        "You have been pre-approved for a personal loan of $10,000. No credit check required.",
        "Investment opportunity: Guaranteed 30% returns. Send us your deposit to get started.",
        "Earn money fast by completing simple online surveys. Sign up now for free.",

        # Generic urgency phishing
        "URGENT: Update your billing information to avoid service suspension. Click the link below.",
        "Action Required: Your email quota is full. Upgrade now or lose access to your inbox.",
        "Your account will be permanently deleted in 24 hours unless you verify your email here.",
        "This is a final warning. Respond immediately to avoid account termination.",
        "Alert: Your password expires today. Click here to keep your current password.",
        "Verify your account now to prevent unauthorized access and protect your data.",

        # Malware / Attachment lures
        "Please review the attached document regarding your recent transaction.",
        "Your invoice is attached. Please open and confirm the payment details urgently.",
        "Your account statement is ready. Download the file attached to view your balance.",

        # --- HAM (SAFE) EXAMPLES ---
        # Work / Professional
        "Hey Bob, are we still on for lunch tomorrow?",
        "Please find the attached invoice for last month's services.",
        "Let's schedule a meeting for next Tuesday to discuss the project timeline.",
        "Can you send over the Q3 report by end of day? Thanks.",
        "I reviewed the pull request, it looks good. Go ahead and merge it.",
        "Don't forget to submit your timesheets before the end of the week.",
        "Here are the meeting notes from yesterday's team discussion.",
        "Could you update the presentation slides before the client call on Thursday?",
        "The budget proposal has been approved. We can proceed with the new hires.",
        "The server deployment is scheduled for Saturday at 2 AM. Please be on standby.",
        "Just a reminder that the performance reviews are due at the end of the month.",
        "Can you share the access credentials for the staging environment?",
        "I've updated the project plan. Please review and add your comments.",
        "Welcome to the team! Your onboarding materials are attached.",
        "Your leave request for next week has been approved.",

        # Personal / Social
        "Happy birthday! Hope you have a fantastic day, let's celebrate soon!",
        "The team outing is confirmed for Friday at 6 PM. Do you have dietary restrictions?",
        "Are you available for a quick call this afternoon? I wanted to catch up.",
        "Great game last night! Our team really turned it around in the second half.",
        "Can you recommend a good restaurant near downtown? We're planning a dinner.",
        "I'll be in town next weekend, would love to grab coffee if you're free.",
        "Did you see the news about the local elections? Quite surprising results.",
        "Looking forward to seeing you at the conference next month.",

        # Informational / Newsletters (legitimate)
        "Your monthly bank statement for April is now available in your online banking portal.",
        "This is a confirmation that your password was successfully changed on May 1st.",
        "Your order #78432 has shipped and will arrive by Friday. Track your package here.",
        "Thank you for your donation to the charity. Your receipt is attached for your records.",
        "Reminder: Your dentist appointment is scheduled for Monday at 10 AM.",
        "Your GitHub repository has 3 new pull request comments. Review them at your convenience.",
        "Weekly digest: Here are the top news stories from this week.",
        "Your annual subscription has been renewed successfully. No action is required.",
        "The project deadline has been extended by one week. Please update your tasks accordingly.",
        "I wanted to follow up on the proposal I sent last Thursday. Any feedback?",
        "Your flight booking confirmation for June 15th is attached. Safe travels!",
        "Just saw your message, give me a few minutes and I'll call you back.",

        # --- PROMOTIONAL EXAMPLES (label=2) ---
        # Tech / SaaS marketing
        "Introducing our new AI platform. From idea to production, all under one platform. Register here.",
        "Join us at the conference this week. Visit our booth and see a live demo of our product.",
        "Announcing the new version of our app with faster performance and a redesigned UI. Update now.",
        "You are invited to our exclusive webinar on AI trends. Reserve your free spot today.",
        "Our biggest sale of the year is here! Get 50% off all plans until Sunday. Shop now.",
        "New feature alert! You can now integrate our tool with Slack. Read the full release notes.",
        "We are hiring! Join our growing team as a Senior Engineer. Apply by the end of the month.",
        "Your free trial is ending soon. Upgrade to a paid plan to keep your data and features.",
        "Discover how top companies use our platform to ship faster. Read the case study.",
        "The Archie platform launches this week at eMerge Americas in Miami. Register to attend.",

        # E-commerce / Retail marketing
        "Black Friday is here! Up to 70% off on all electronics. Deals end at midnight.",
        "Your wishlist item is back in stock! Order now before it sells out again.",
        "Exclusive member offer: Get free shipping on your next three orders. No code needed.",
        "Flash sale: 24 hours only. Use code SAVE20 at checkout for 20% off your entire cart.",
        "New arrivals just dropped. Check out the latest collection and find your new favourite.",

        # Newsletter / Content marketing
        "This week in tech: the biggest stories, tools, and tips you need to know.",
        "Your weekly digest is ready. Here are the top 5 articles our community loved this week.",
        "We published a new blog post on how to improve your team's productivity. Read it now.",
        "Monthly product update: here is everything we shipped in April. Thank you for your support.",
        "Our founder shares his story on the podcast this week. Listen to the episode here.",

        # Event / Community invites
        "You are invited to our annual user conference. Early bird tickets are now available.",
        "Join thousands of professionals at our online summit next month. Secure your free seat.",
        "Our community meetup is happening in your city next Friday. RSVP to save your spot.",
        "We are hosting a free live coding session this Thursday at 3 PM. Register to join.",
        "Join our beta program and get early access to features before they go public.",

        # Transactional-style marketing
        "You have 500 reward points about to expire. Redeem them before they are gone.",
        "Based on your browsing, we think you will love these new arrivals. Take a look.",
        "Your friends are already using our app! Invite them and earn a free month for both of you.",
        "We miss you! Come back and see what is new. Here is a 15% discount just for you.",
        "Complete your profile to unlock personalized recommendations and exclusive content.",

        # Conference / Event announcements
        "Albert Santalo takes the AI Stage at eMerge Americas. Swing by Booth 221 to meet the team.",
        "Save the date: our annual product keynote is on June 10th. Stream it live for free.",
        "You are registered for the AI Masterclass on April 22nd. Here are the session details.",
        "The waitlist for our new tool is now open. Sign up to be the first to get access.",
        "Limited spots remaining for our workshop next week. Reserve yours before they run out.",
    ]
    labels = (
        [LABEL_PHISHING]    * 35 +
        [LABEL_SAFE]        * 35 +
        [LABEL_PROMOTIONAL] * 35
    )
    
    preprocessor = NLTKPreprocessor()
    cleaned_texts = [preprocessor.clean(t) for t in texts]
    
    extractor = TfidfExtractor()
    extractor.fit(cleaned_texts)
    features = extractor.transform(cleaned_texts)
    
    classifier = LogisticRegressionClassifier()
    classifier.set_feature_names(extractor.get_feature_names())
    classifier.train(features, labels)
    
    repo = JoblibModelRepository()
    os.makedirs("models", exist_ok=True)
    repo.save_extractor(extractor, "models/extractor.pkl")
    repo.save_classifier(classifier, "models/classifier.pkl")
    print("Dummy model trained and saved to 'models/' directory.")

def main():
    parser = argparse.ArgumentParser(description="Phishing Detection System API")
    parser.add_argument("--train", action="store_true", help="Train a baseline model")
    parser.add_argument("--serve", action="store_true", help="Run the API web server")
    
    args = parser.parse_args()
    
    if args.train:
        train_dummy_model()
        return

    if args.serve:
        try:
            repo = JoblibModelRepository()
            extractor = repo.load_extractor("models/extractor.pkl")
            classifier = repo.load_classifier("models/classifier.pkl")
            preprocessor = NLTKPreprocessor()
            
            analyzer = PhishingAnalyzer(preprocessor, extractor, classifier)
            app = create_app(analyzer)
            
            print("Listening on http://0.0.0.0:5000 ...")
            app.run(host="0.0.0.0", port=5000)
        except Exception as e:
            print(f"Error: {e}")
            print("Action: Please run 'python main.py --train' first to generate model files.")

if __name__ == "__main__":
    main()
