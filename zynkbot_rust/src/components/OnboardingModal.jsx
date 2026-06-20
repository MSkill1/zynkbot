import React, { useState, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import '../styles/OnboardingModal.css';

export default function OnboardingModal({ isOpen, onClose, userId }) {
  const [currentStep, setCurrentStep] = useState(0);
  const [userResponses, setUserResponses] = useState({});
  const [currentInput, setCurrentInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [summary, setSummary] = useState(null);
  const contentRef = useRef(null);

  // Reset all state each time the modal opens
  useEffect(() => {
    if (isOpen) {
      setCurrentStep(0);
      setUserResponses({});
      setCurrentInput('');
      setSummary(null);
    }
  }, [isOpen]);

  // Scroll content area back to top on each new question
  useEffect(() => {
    if (contentRef.current) {
      contentRef.current.scrollTop = 0;
    }
  }, [currentStep]);

  const questions = [
    {
      id: 'intro',
      text: "Hi! I'm Zynkbot - think of me as a companion who remembers everything you share.\n\nBefore we start, know this: everything you tell me is stored as memories that you can edit, delete, or update anytime. I only know what you choose to share, and you're in complete control.\n\nReady to help me get to know you?",
      placeholder: "Type 'yes' or 'ready' to begin...",
      isIntro: true
    },
    {
      id: 'name_age',
      text: "Let's start with your name and age. What's your full name, what do you go by day-to-day, and how old are you?",
      placeholder: "My full name is... I go by... I'm X years old."
    },
    {
      id: 'family',
      text: "Tell me about your family. Who are the important people in your life? Parents, siblings, children - whoever matters to you.",
      placeholder: "Tell me about your family..."
    },
    {
      id: 'work',
      text: "What do you do for work, or what takes up most of your time? Feel free to say as much or as little as you want.",
      placeholder: "I work as..."
    },
    {
      id: 'interests',
      text: "What do you care about? What gets you excited or makes you happy? Could be hobbies, passions, causes - whatever lights you up.",
      placeholder: "What do you care about..."
    },
    {
      id: 'goals',
      text: "What are you working on or hoping to achieve right now? Dreams, projects, challenges you're facing - I'm here to remember and support you.",
      placeholder: "What are you working towards..."
    },
    {
      id: 'purpose',
      text: "Last one: What do you hope I can help you with? Why did you decide to try me out?",
      placeholder: "I hope you can help me with..."
    }
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!currentInput.trim()) return;

    const currentQuestion = questions[currentStep];

    const updatedResponses = {
      ...userResponses,
      [currentQuestion.id]: currentInput
    };

    setUserResponses(updatedResponses);
    setIsProcessing(true);

    try {
      if (!currentQuestion.isIntro) {
        await invoke('store_onboarding_response', {
          userId: userId,
          questionId: currentQuestion.id,
          question: currentQuestion.text,
          answer: currentInput
        });
      }

      if (currentStep < questions.length - 1) {
        setCurrentStep(currentStep + 1);
        setCurrentInput('');
      } else {
        const summaryText = await invoke('complete_onboarding', {
          userId: userId,
          responses: updatedResponses
        });
        setSummary(summaryText);
      }
    } catch (error) {
      console.error('Error storing onboarding response:', error);
      alert(`Error: ${error}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleExit = () => {
    if (window.confirm('Exit onboarding? You can always run it later from the sidebar. Anything you\'ve answered so far has already been saved.')) {
      onClose();
    }
  };

  const handleFinish = () => {
    onClose();
  };

  if (!isOpen) return null;

  const currentQuestion = questions[currentStep];
  const progress = Math.round((currentStep / (questions.length - 1)) * 100);

  return (
    <div className="onboarding-modal-overlay">
      <div className="onboarding-modal">
        <div className="onboarding-header">
          <h2>🎯 Get to Know You</h2>
          {currentStep > 0 && !summary && (
            <button className="onboarding-close" onClick={handleExit}>Exit</button>
          )}
        </div>

        {!summary ? (
          <>
            <div className="onboarding-progress">
              <div className="onboarding-progress-bar" style={{ width: `${progress}%` }} />
              <span className="onboarding-progress-text">
                {currentStep === 0 ? 'Welcome' : `Question ${currentStep} of ${questions.length - 1}`}
              </span>
            </div>

            <div className="onboarding-content" ref={contentRef}>
              <div className="onboarding-question">
                {currentQuestion.text.split('\n').map((line, i) => (
                  <p key={i}>{line}</p>
                ))}
              </div>

              {currentStep === 0 ? (
                <div className="onboarding-actions onboarding-intro-actions">
                  <button className="onboarding-cancel" onClick={onClose}>Not Now</button>
                  <button className="onboarding-submit" onClick={() => setCurrentStep(1)}>Let's Go</button>
                </div>
              ) : (
                <form onSubmit={handleSubmit} className="onboarding-form">
                  <textarea
                    key={currentStep}
                    value={currentInput}
                    onChange={(e) => setCurrentInput(e.target.value)}
                    placeholder={currentQuestion.placeholder}
                    rows={4}
                    disabled={isProcessing}
                    autoFocus
                  />

                  <div className="onboarding-actions">
                    <button
                      type="submit"
                      disabled={!currentInput.trim() || isProcessing}
                      className="onboarding-submit"
                    >
                      {isProcessing ? 'Saving...' : currentStep === questions.length - 1 ? 'Finish' : 'Next'}
                    </button>
                  </div>
                </form>
              )}
            </div>
          </>
        ) : (
          <div className="onboarding-summary">
            <h3>Thank you for sharing that with me! 🌟</h3>
            <div className="onboarding-summary-content">
              {summary}
            </div>
            <div className="onboarding-reminder">
              <strong>Tip:</strong> You can edit or delete any of these in the Memory Manager anytime — just open it from the sidebar and filter by the "onboarding" namespace.
            </div>
            <button onClick={handleFinish} className="onboarding-finish">
              Start Chatting
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
