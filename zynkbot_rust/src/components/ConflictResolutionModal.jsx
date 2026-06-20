import React, { useState } from "react";
import "../styles/ConflictResolutionModal.css";

export default function ConflictResolutionModal({ isOpen, conflict, onResolve, onClose }) {
  const [selectedOption, setSelectedOption] = useState(null);
  const [explanation, setExplanation] = useState("");

  if (!isOpen || !conflict) return null;

  const { memoryA, memoryB, sharedEntities } = conflict;

  const memoryADate = new Date(memoryA.created_at);
  const memoryBDate = new Date(memoryB.created_at);
  const isMemoryAOlder = memoryADate < memoryBDate;

  const oldMemory = isMemoryAOlder ? memoryA : memoryB;
  const newMemory = isMemoryAOlder ? memoryB : memoryA;

  const keepOldValue = isMemoryAOlder ? "memoryA" : "memoryB";
  const keepNewValue = isMemoryAOlder ? "memoryB" : "memoryA";

  const handleOptionChange = (option) => {
    setSelectedOption(option);
    if (option !== "both_with_explanation") {
      setExplanation("");
    }
  };

  const handleConfirm = () => {
    if (!selectedOption) {
      alert("Please select how to resolve this conflict");
      return;
    }
    if (selectedOption === "both_with_explanation" && !explanation.trim()) {
      alert("Please provide an explanation, or choose a different option");
      return;
    }
    onResolve({
      option: selectedOption,
      memoryA,
      memoryB,
      explanation: selectedOption === "both_with_explanation" ? explanation : null,
    });
    setSelectedOption(null);
    setExplanation("");
  };

  const handleCancel = () => {
    setSelectedOption(null);
    setExplanation("");
    onClose();
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content conflict-resolution">
        <button className="modal-close" onClick={handleCancel}>×</button>

        <h2>⚠️ Possible Conflict Detected</h2>
        <p className="conflict-subtitle">
          Two memories may contradict each other
          {sharedEntities && sharedEntities.length > 0 && <> about <strong>{sharedEntities.join(", ")}</strong></>}.
          Choose how to resolve it.
        </p>

        <div className="conflict-layout">
          {/* Left column: memory boxes */}
          <div className="conflict-left">
            <h3>Conflicting Memories</h3>
            <div className="conflict-memories">
              <div className="conflict-memory memory-a">
                <div className="memory-header">
                  <span className="memory-label">Memory #{oldMemory.id} <strong>(OLD)</strong></span>
                  {oldMemory.title && <span className="memory-title">{oldMemory.title}</span>}
                </div>
                <div className="memory-content">{oldMemory.content}</div>
              </div>

              <div className="conflict-divider">
                <span className="vs-text">VS</span>
              </div>

              <div className="conflict-memory memory-b">
                <div className="memory-header">
                  <span className="memory-label"><strong>(NEW)</strong></span>
                  {newMemory.title && <span className="memory-title">{newMemory.title}</span>}
                </div>
                <div className="memory-content">{newMemory.content}</div>
              </div>
            </div>
          </div>

          {/* Right column: resolution options */}
          <div className="conflict-right">
            <div className="resolution-options">
              <h3>How should I handle this?</h3>

              {/* Option 1: Keep old */}
              <label className="resolution-option">
                <input
                  type="radio"
                  name="resolution"
                  value={keepOldValue}
                  checked={selectedOption === keepOldValue}
                  onChange={() => handleOptionChange(keepOldValue)}
                />
                <span className="option-label">
                  <strong>Keep old memory #{oldMemory.id}</strong>
                  <span className="option-description">Discard the new statement — the older memory is correct</span>
                </span>
              </label>

              {/* Option 2: Keep new */}
              <label className="resolution-option">
                <input
                  type="radio"
                  name="resolution"
                  value={keepNewValue}
                  checked={selectedOption === keepNewValue}
                  onChange={() => handleOptionChange(keepNewValue)}
                />
                <span className="option-label">
                  <strong>Keep new memory</strong>
                  <span className="option-description">Discard the older memory — the new statement replaces it</span>
                </span>
              </label>

              {/* Option 3: Not a contradiction */}
              <label className="resolution-option">
                <input
                  type="radio"
                  name="resolution"
                  value="not_a_contradiction"
                  checked={selectedOption === "not_a_contradiction"}
                  onChange={() => handleOptionChange("not_a_contradiction")}
                />
                <span className="option-label">
                  <strong>Not a contradiction</strong>
                  <span className="option-description">Both are true and unrelated — keep both, remove the contradiction edge</span>
                </span>
              </label>

              {/* Option 4: Accept contradiction */}
              <label className="resolution-option">
                <input
                  type="radio"
                  name="resolution"
                  value="keep_both"
                  checked={selectedOption === "keep_both"}
                  onChange={() => handleOptionChange("keep_both")}
                />
                <span className="option-label">
                  <strong>Accept the contradiction</strong>
                  <span className="option-description">Keep both memories with the contradiction edge intact</span>
                </span>
              </label>

              {selectedOption === "keep_both" && (
                <div className="contradiction-warning">
                  <strong>⚠️ Heads up</strong>
                  <p>
                    Both memories will be kept with a contradiction edge (red arrow) in your memory graph.
                    This may cause hallucinations in future responses when both memories are recalled together.
                    You can return to resolve this conflict at any time in the Memory Manager.
                  </p>
                </div>
              )}

              {/* Option 5: Resolve with explanation */}
              <label className="resolution-option">
                <input
                  type="radio"
                  name="resolution"
                  value="both_with_explanation"
                  checked={selectedOption === "both_with_explanation"}
                  onChange={() => handleOptionChange("both_with_explanation")}
                />
                <span className="option-label">
                  <strong>Resolve with explanation</strong>
                  <span className="option-description">Both are true — explain why, and store it as a resolving memory</span>
                </span>
              </label>

              {selectedOption === "both_with_explanation" && (
                <div className="explanation-container">
                  <div className="explanation-help">
                    <strong>Explain how both can be true:</strong>
                    <p>
                      Your explanation will be stored as a new memory with a <em>resolves</em> relationship
                      to both, so I understand the full picture in the future.
                    </p>
                  </div>
                  <textarea
                    className="explanation-input"
                    placeholder="Example: I have two dogs — Max is my golden retriever and Wendy is my new puppy"
                    value={explanation}
                    onChange={(e) => setExplanation(e.target.value)}
                    rows={4}
                  />
                  <div className="explanation-note">
                    💡 Be specific about the context that makes both memories true
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="modal-actions">
          <button
            className="btn-confirm"
            onClick={handleConfirm}
            disabled={!selectedOption || (selectedOption === "both_with_explanation" && !explanation.trim())}
          >
            Confirm Resolution
          </button>
          <button className="btn-cancel" onClick={handleCancel}>
            Cancel
          </button>
        </div>

        <div className="conflict-note">
          <strong>Note:</strong> This helps me learn and remember correctly. Your choice will be recorded
          and I'll reference it in the future.
        </div>
      </div>
    </div>
  );
}
