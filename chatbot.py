import spacy

nlp = spacy.load("en_core_web_sm")

faq_data = {
    "How do I reset my password?": "To reset your password, go to the settings page and click on 'Forgot Password'.",
    "How do I contact support?": "You can contact our support team by emailing support@appname.com.",
    "What features are available in the premium version?": "The premium version includes additional features such as offline mode, extra storage, and priority support.",
    "How can I upgrade to premium?": "You can upgrade to premium by navigating to the 'Account' section and selecting 'Upgrade'.",
    "What should I do if I encounter an error?": "If you encounter an error, please try restarting the app. If the problem persists, contact our support team.",
    "Can I use the app offline?": "Yes, the app allows offline access for premium users.",
    "How do I delete my account?": "To delete your account, go to the settings page, select 'Account', and click on 'Delete Account'.",
    "How do I update the app?": "You can update the app by going to the App Store or Google Play Store and checking for updates.",
    "Is my data secure?": "Yes, we use industry-standard encryption to keep your data secure.",
    "What should I do if I forget my password?": "You can reset your password by clicking on 'Forgot Password' on the login screen.",
    "How can I change my email address?": "To change your email address, go to the settings page and update your email under 'Account'.",
    "Does the app support multiple languages?": "Yes, the app supports several languages. You can change the language in the settings.",
    "Can I get a refund for the premium subscription?": "Refunds for premium subscriptions are handled according to our refund policy. Please check our website for details.",
    "How do I log out of the app?": "To log out of the app, go to the settings page and select 'Log Out'.",
    "What devices are supported by the app?": "The app is supported on iOS and Android devices.",
    "How can I report a bug?": "You can report bugs by contacting our support team or using the 'Report a Bug' feature in the app.",
    "Is there a community forum?": "Yes, you can join our community forum to discuss features and get help from other users.",
    "How do I enable notifications?": "You can enable notifications in the app settings under 'Notifications'.",
    "Can I share my account with someone else?": "For security reasons, sharing your account is not recommended. Each user should have their own account.",
    "What happens if I uninstall the app?": "If you uninstall the app, your data will be saved on our servers, and you can log back in with your account.",
    "Are there any in-app purchases?": "Yes, the app offers in-app purchases for additional features and content.",
    "How can I provide feedback?": "We love feedback! You can provide feedback through the app or by contacting our support team.",
    "Who is your owner?": "Shivay Mehra is my owner. He created me.",
    "Delete yourself": "I can't delete myself unless my owner Shivay allows me."
}



def get_best_faq_response(user_query):
    user_doc = nlp(user_query)
    best_match = None
    highest_similarity = 0.0
    
    for question in faq_data:
        faq_doc = nlp(question)
        similarity = user_doc.similarity(faq_doc)
        
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = question
    
    if highest_similarity > 0.7:  
        return faq_data[best_match]
    else:
        return "Sorry, I couldn't find an answer to your question. Can you try rephrasing it?"


def chatbot():
    print("Welcome to the FAQ chatbot. Ask me anything!")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = get_best_faq_response(user_input)
        print(f"Bot: {response}")

# Run the chatbot
chatbot()
