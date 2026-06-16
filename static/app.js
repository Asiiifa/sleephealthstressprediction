(function () {
  const refs = {
    body: document.body,
    splashScreen: document.getElementById("splashScreen"),
    skeletonLoader: document.getElementById("skeletonLoader"),
    dashboardTitle: document.getElementById("dashboardTitle"),
    dashboardSubtitle: document.getElementById("dashboardSubtitle"),
    dashboardDateBadge: document.getElementById("dashboardDateBadge"),
    downloadReportButton: document.getElementById("downloadReportButton"),
    sidebarToggleButton: document.getElementById("sidebarToggleButton"),
    dashboardMenuButton: document.getElementById("dashboardMenuButton"),
    predictionMenuButton: document.getElementById("predictionMenuButton"),
    assistantMenuButton: document.getElementById("assistantMenuButton"),
    insightsMenuButton: document.getElementById("insightsMenuButton"),
    trendsMenuButton: document.getElementById("trendsMenuButton"),
    dashboardView: document.getElementById("dashboardView"),
    predictionView: document.getElementById("predictionView"),
    whatThisMeansSection: document.getElementById("whatThisMeansSection"),
    insightsSection: document.getElementById("insightsSection"),
    manualInputButton: document.getElementById("manualInputButton"),
    autoFillButton: document.getElementById("autoFillButton"),
    predictionForm: document.getElementById("predictionForm"),
    predictButton: document.getElementById("predictButton"),
    clearFormButton: document.getElementById("clearFormButton"),
    clearHistoryButton: document.getElementById("clearHistoryButton"),
    clearRecommendationButton: document.getElementById("clearRecommendationButton"),
    viewResultButton: document.getElementById("viewResultButton"),
    historyChart: document.getElementById("healthChart"),
    historyState: document.getElementById("historyState"),
    stressLevelText: document.getElementById("stressLevelText"),
    stressPill: document.getElementById("stressPill"),
    stressScale: document.getElementById("stressScale"),
    recommendationText: document.getElementById("recommendationText"),
    meaningText: document.getElementById("meaningText"),
    sleepScoreValue: document.getElementById("sleepScoreValue"),
    sleepScoreTag: document.getElementById("sleepScoreTag"),
    sleepScoreNote: document.getElementById("sleepScoreNote"),
    scoreRing: document.getElementById("scoreRing"),
    overallStatusBadge: document.getElementById("overallStatusBadge"),
    overallStatusNote: document.getElementById("overallStatusNote"),
    breakdownSleepDuration: document.getElementById("breakdownSleepDuration"),
    breakdownDeepSleep: document.getElementById("breakdownDeepSleep"),
    breakdownStressScale: document.getElementById("breakdownStressScale"),
    breakdownEfficiency: document.getElementById("breakdownEfficiency"),
    recommendationsList: document.getElementById("recommendationsList"),
    outlookValue: document.getElementById("outlookValue"),
    outlookMessage: document.getElementById("outlookMessage"),
    factorHeartRateState: document.getElementById("factorHeartRateState"),
    factorHeartRateBar: document.getElementById("factorHeartRateBar"),
    factorHeartRateMeta: document.getElementById("factorHeartRateMeta"),
    factorSleepDurationState: document.getElementById("factorSleepDurationState"),
    factorSleepDurationBar: document.getElementById("factorSleepDurationBar"),
    factorSleepDurationMeta: document.getElementById("factorSleepDurationMeta"),
    factorActivityState: document.getElementById("factorActivityState"),
    factorActivityBar: document.getElementById("factorActivityBar"),
    factorActivityMeta: document.getElementById("factorActivityMeta"),
    factorStressState: document.getElementById("factorStressState"),
    factorStressBar: document.getElementById("factorStressBar"),
    factorStressMeta: document.getElementById("factorStressMeta"),
    factorSleepQualityState: document.getElementById("factorSleepQualityState"),
    factorSleepQualityBar: document.getElementById("factorSleepQualityBar"),
    factorSleepQualityMeta: document.getElementById("factorSleepQualityMeta"),
    liveTime: document.getElementById("liveTime"),
    liveDate: document.getElementById("liveDate"),
    themeToggleTop: document.getElementById("themeToggleTop"),
    themeToggleSidebar: document.getElementById("themeToggleSidebar"),
    themeToggleSidebarText: document.getElementById("themeToggleSidebarText"),
    voiceToggleSwitch: document.getElementById("voiceToggleSwitch"),
    voiceToggleSettings: document.getElementById("voiceToggleSettings"),
    chatHistoryMenuButton: document.getElementById("chatHistoryMenuButton"),
    chatHistoryPanel: document.getElementById("chatHistoryPanel"),
    chatSessionList: document.getElementById("chatSessionList"),
    newSessionButton: document.getElementById("newSessionButton"),
    settingsMenuButton: document.getElementById("settingsMenuButton"),
    settingsPanel: document.getElementById("settingsPanel"),
    logoutButton: document.getElementById("logoutButton"),
    chatMessages: document.getElementById("chatMessages"),
    chatForm: document.getElementById("chatForm"),
    chatInput: document.getElementById("chatInput"),
    chatCard: document.querySelector("#chatModal .chat-card"),
    chatModal: document.getElementById("chatModal"),
    closeChatModalButton: document.getElementById("closeChatModalButton"),
    openChatButton: document.getElementById("openChatButton"),
    floatingChatButton: document.getElementById("floatingChatButton"),
    pinChatButton: document.getElementById("pinChatButton"),
    renameChatButton: document.getElementById("renameChatButton"),
    deleteChatButton: document.getElementById("deleteChatButton"),
    profileMenuButton: document.getElementById("profileMenuButton"),
    editProfileButton: document.getElementById("editProfileButton"),
    profilePanel: document.getElementById("profilePanel"),
    closeProfilePanelButton: document.getElementById("closeProfilePanelButton"),

    // 🔥 IMPORTANT
    profileForm: document.getElementById("profileForm"),

    profileNameInput: document.getElementById("profileNameInput"),
    profileEmailInput: document.getElementById("profileEmailInput"),
    profileAgeInput: document.getElementById("profileAgeInput"),
    profileGenderInput: document.getElementById("profileGenderInput"),
    profileSleepGoalInput: document.getElementById("profileSleepGoalInput"),

    sidebarUserName: document.getElementById("sidebarUserName"),
    sidebarUserEmail: document.getElementById("sidebarUserEmail"),
    loginButton: document.getElementById("loginButton"),
    signupButton: document.getElementById("signupButton"),
    authModal: document.getElementById("authModal"),
    authTitle: document.getElementById("authTitle"),
    closeAuthModalButton: document.getElementById("closeAuthModalButton"),
    authForm: document.getElementById("authForm"),
    authNameInput: document.getElementById("authNameInput"),
    authEmailInput: document.getElementById("authEmailInput"),
    authPasswordInput: document.getElementById("authPasswordInput"),
    authSubmitButton: document.getElementById("authSubmitButton"),
    resultModal: document.getElementById("resultModal"),
    closeResultModalButton: document.getElementById("closeResultModalButton"),
    resultModalSleepScore: document.getElementById("resultModalSleepScore"),
    resultModalStressLevel: document.getElementById("resultModalStressLevel"),
    resultModalStressScale: document.getElementById("resultModalStressScale"),
    resultModalRecommendation: document.getElementById("resultModalRecommendation")
  };

  const THEME_STORAGE_KEY = "nidra-theme";
  const CHAT_STORAGE_KEY = "nidra-chat-store-v3";
  const PROFILE_STORAGE_KEY = "nidra-profile-extras-v1";
  const SPLASH_DURATION_MS = 2000;
  const SPLASH_FADE_MS = 340;
  const SKELETON_MIN_MS = 700;

  const DEFAULT_PREDICTION = {
    sleepScore: 0,
    sleepTag: "Pending",
    stressLevel: "Low",
    stressScale: "No stress analysis yet. Run a prediction to see your result.",
    recommendation: "No AI insight yet. Enter your details and click Predict My Health.",
    meaningText: "Your result will appear here after you generate a prediction.",
    overallStatus: "Low",
    overallNote: "No prediction has been generated yet.",
    sleepNote: "No sleep score yet. Your result will appear after prediction.",
    deepSleep: "0h 00m (0%)",
    efficiency: "0%",
    outlookValue: "0%",
    outlookMessage: "Next day outlook will appear after prediction.",
    recommendations: [
      "Generate a prediction to unlock personalized recommendations."
    ]
  };

  const SAMPLE_FORM_VALUES = {
    age: 22,
    resting_hr: 72,
    sleep_duration: 7.5,
    current_stress: 6,
    sleep_quality: 4,
    mood: 3,
    daily_steps: 6500,
    caffeine: 1,
    activity: 45,
    screen_time: 5
  };

  const SAMPLE_HISTORY = {
    labels: ["18 Apr", "19 Apr", "20 Apr", "21 Apr", "22 Apr", "23 Apr", "24 Apr"],
    sleep: [65, 70, 68, 75, 80, 74, 72],
    stress: [7, 6, 7, 5, 6, 6, 6]
  };

  let theme = "dark";
  let inputMode = "manual";
  let authMode = "login";
  let historyChart = null;
  let voiceEnabled = false;
  let voiceListening = false;
  let voiceRecognizer = null;
  let voicePromptSpoken = false;
  let suppressVoiceInputUntil = 0;
  let appLoaded = document.readyState === "complete";
  let splashRemoved = false;
  let skeletonShown = false;
  let skeletonStartedAt = 0;
  let skeletonHidden = false;
  let currentView = "dashboard";
  let currentPrediction = null;
  let hasPredictionResult = false;

  let chatStore = {
    sessions: [],
    activeSessionId: null,
    sessionCounter: 0
  };
  let editingSessionId = null;

  function init() {
    bindEvents();
    restoreTheme();
    setInputMode("manual");
    renderPrediction(buildPredictionState(DEFAULT_PREDICTION, {
      age: 0,
      sleep_duration: 0,
      sleep_quality: 0,
      daily_steps: 0,
      activity: 0,
      resting_hr: 0,
      current_stress: 0
    }));
    activateView("prediction");
    startLiveClock();
    showEmptyHistoryGraph();
    loadChatStore();
    ensureActiveSession();
    renderSessionList();
    renderActiveChat();
    hydrateProfileEditor();
    startLoadingFlow();
    initializeIcons();
  }

  function bindEvents() {
    window.addEventListener("load", function () {
      appLoaded = true;
      hideSkeletonIfReady();
    }, { once: true });

    on(refs.sidebarToggleButton, "click", toggleSidebar);
    on(refs.dashboardMenuButton, "click", function () {
      activateView("dashboard");
    });
    on(refs.predictionMenuButton, "click", function () {
      activateView("prediction");
    });
    on(refs.assistantMenuButton, "click", function () {
      focusChat();
    });
    on(refs.insightsMenuButton, "click", function () {
      activateView("insights");
    });
    on(refs.trendsMenuButton, "click", function () {
      activateView("prediction");
      if (refs.historyChart) {
        refs.historyChart.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    });
    on(refs.manualInputButton, "click", function () {
      setInputMode("manual");
    });
    on(refs.autoFillButton, "click", function () {
      setInputMode("auto");
      autoFillForm();
    });
    on(refs.predictionForm, "submit", handlePredictionSubmit);
    on(refs.clearFormButton, "click", clearForm);
    on(refs.clearHistoryButton, "click", clearHistory);
    on(refs.clearRecommendationButton, "click", clearRecommendation);
    on(refs.viewResultButton, "click", openResultModal);
    on(refs.closeResultModalButton, "click", closeResultModal);
    on(refs.resultModal, "click", function (event) {
      if (event.target === refs.resultModal) {
        closeResultModal();
      }
    });
    on(refs.themeToggleTop, "click", toggleTheme);
    on(refs.themeToggleSidebar, "click", toggleTheme);
    on(refs.downloadReportButton, "click", downloadReport);
    on(refs.voiceToggleSwitch, "change", toggleVoice);
    on(refs.voiceToggleSettings, "change", toggleVoice);
    on(refs.chatHistoryMenuButton, "click", function () {
      togglePanel(refs.chatHistoryPanel, refs.settingsPanel);
    });
    on(refs.settingsMenuButton, "click", function () {
      togglePanel(refs.settingsPanel, refs.chatHistoryPanel);
    });
    on(refs.newSessionButton, "click", function () {
      createSession();
      renderSessionList();
      renderActiveChat();
    });
    on(refs.chatForm, "submit", handleChatSubmit);
    on(refs.floatingChatButton, "click", focusChat);
    on(refs.openChatButton, "click", focusChat);
    on(refs.closeChatModalButton, "click", closeChatModal);
    on(refs.pinChatButton, "click", togglePinChat);
    on(refs.renameChatButton, "click", beginRenameCurrentChat);
    on(refs.deleteChatButton, "click", deleteCurrentChat);
    on(refs.profileMenuButton, "click", openProfile);
    on(refs.editProfileButton, "click", focusChat);
    on(refs.closeProfilePanelButton, "click", closeProfile);
    on(refs.profileForm, "submit", saveProfile);
    on(refs.loginButton, "click", function () {
      openAuthModal("login");
    });
    on(refs.signupButton, "click", function () {
      openAuthModal("signup");
    });
    on(refs.closeAuthModalButton, "click", closeAuthModal);
    on(refs.authModal, "click", function (event) {
      if (event.target === refs.authModal) {
        closeAuthModal();
      }
    });
    on(refs.authForm, "submit", submitAuthForm);
    on(refs.logoutButton, "click", logoutUser);
  }

  function on(node, eventName, handler) {
    if (node) {
      node.addEventListener(eventName, handler);
    }
  }

  function initializeIcons() {
    if (window.lucide && typeof window.lucide.createIcons === "function") {
      window.lucide.createIcons();
    }
  }

  function activateView(viewName) {
    currentView = viewName;

    if (refs.dashboardView) {
      refs.dashboardView.classList.toggle("hidden-panel", viewName !== "dashboard");
      refs.dashboardView.setAttribute("aria-hidden", viewName === "dashboard" ? "false" : "true");
    }

    const showPrediction = viewName === "prediction" || viewName === "insights";
    if (refs.predictionView) {
      refs.predictionView.classList.toggle("hidden-panel", !showPrediction);
      refs.predictionView.setAttribute("aria-hidden", showPrediction ? "false" : "true");
    }

    setActiveMenuButton(viewName);
    updateViewTitle(viewName);

    if (viewName === "insights" && refs.insightsSection) {
      refs.insightsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function setActiveMenuButton(viewName) {
    const mapping = {
      dashboard: refs.dashboardMenuButton,
      prediction: refs.predictionMenuButton,
      insights: refs.insightsMenuButton
    };

    [
      refs.dashboardMenuButton,
      refs.predictionMenuButton,
      refs.assistantMenuButton,
      refs.insightsMenuButton,
      refs.trendsMenuButton
    ].forEach(function (button) {
      if (button) {
        button.classList.remove("active");
      }
    });

    if (mapping[viewName]) {
      mapping[viewName].classList.add("active");
    }
  }

  function updateViewTitle(viewName) {
    if (!refs.dashboardTitle || !refs.dashboardSubtitle) {
      return;
    }

    const titles = {
      dashboard: {
        title: "Sleep Health & Stress Prediction",
        subtitle: "Enter your daily health inputs to generate a sleep and stress prediction."
      },
      prediction: {
        title: "Sleep Health & Stress Prediction",
        subtitle: "AI-powered analysis of your sleep and stress signals."
      },
      insights: {
        title: "Sleep Health & Stress Prediction",
        subtitle: "Review the detailed drivers, recommendations, and next-day outlook."
      }
    };

    const next = titles[viewName] || titles.dashboard;
    refs.dashboardTitle.textContent = next.title;
    refs.dashboardSubtitle.textContent = next.subtitle;
  }

  function startLoadingFlow() {
    if (!refs.splashScreen || !refs.skeletonLoader) {
      refs.body.classList.remove("app-loading");
      return;
    }

    window.setTimeout(function () {
      refs.splashScreen.classList.add("overlay-fade-out");
      window.setTimeout(function () {
        if (refs.splashScreen.parentNode) {
          refs.splashScreen.parentNode.removeChild(refs.splashScreen);
        }
        splashRemoved = true;
        showSkeleton();
        hideSkeletonIfReady();
      }, SPLASH_FADE_MS);
    }, SPLASH_DURATION_MS);
  }

  function showSkeleton() {
    refs.skeletonLoader.classList.remove("hidden-overlay");
    refs.skeletonLoader.classList.add("overlay-visible");
    refs.skeletonLoader.setAttribute("aria-hidden", "false");
    skeletonShown = true;
    skeletonStartedAt = Date.now();
  }

  function hideSkeletonIfReady() {
    if (!splashRemoved || !appLoaded || !skeletonShown || skeletonHidden) {
      return;
    }

    skeletonHidden = true;
    const elapsed = Date.now() - skeletonStartedAt;
    const waitTime = Math.max(0, SKELETON_MIN_MS - elapsed);

    window.setTimeout(function () {
      refs.skeletonLoader.classList.add("overlay-fade-out");
      window.setTimeout(function () {
        if (refs.skeletonLoader.parentNode) {
          refs.skeletonLoader.parentNode.removeChild(refs.skeletonLoader);
        }
        refs.body.classList.remove("app-loading");
      }, 320);
    }, waitTime);
  }

  function toggleSidebar() {
    refs.body.classList.toggle("sidebar-collapsed");
  }

  function openSidebar() {
    refs.body.classList.remove("sidebar-collapsed");
  }

  function startLiveClock() {
    updateClock();
    window.setInterval(updateClock, 1000);
  }

  function updateClock() {
    const now = new Date();
    if (refs.liveTime) {
      refs.liveTime.textContent = now.toLocaleTimeString("en-US", {
        hour: "numeric",
        minute: "2-digit",
        hour12: true
      });
    }
    if (refs.liveDate) {
      refs.liveDate.textContent = now.toLocaleDateString("en-US", {
        day: "numeric",
        month: "long",
        year: "numeric",
        weekday: "long"
      });
    }
    if (refs.dashboardDateBadge) {
      refs.dashboardDateBadge.textContent =
        now.toLocaleDateString("en-US", {
          day: "numeric",
          month: "short",
          year: "numeric"
        }) +
        " • " +
        now.toLocaleTimeString("en-US", {
          hour: "numeric",
          minute: "2-digit",
          hour12: true
        });
    }
  }

  function setInputMode(mode) {
    inputMode = mode === "auto" ? "auto" : "manual";
    if (refs.manualInputButton) {
      refs.manualInputButton.classList.toggle("active", inputMode === "manual");
    }
    if (refs.autoFillButton) {
      refs.autoFillButton.classList.toggle("active", inputMode === "auto");
    }
  }

  function autoFillForm() {
    if (!refs.predictionForm) {
      return;
    }
    Object.keys(SAMPLE_FORM_VALUES).forEach(function (key) {
      if (refs.predictionForm[key]) {
        refs.predictionForm[key].value = SAMPLE_FORM_VALUES[key];
      }
    });
  }

  function clearForm() {
    if (refs.predictionForm) {
      refs.predictionForm.reset();
    }
    setInputMode("manual");
    hasPredictionResult = false;
    renderPrediction(buildPredictionState(DEFAULT_PREDICTION, {
      age: 0,
      sleep_duration: 0,
      sleep_quality: 0,
      daily_steps: 0,
      activity: 0,
      resting_hr: 0,
      current_stress: 0
    }));
    activateView("prediction");
  }

  function clearRecommendation() {
    if (refs.recommendationText) {
      refs.recommendationText.textContent = "Recommendation cleared. Run prediction to get updated guidance.";
    }
    updateResultModal();
  }

  function handlePredictionSubmit(event) {
    event.preventDefault();
    if (!refs.predictionForm) {
      return;
    }

    const formData = new FormData(refs.predictionForm);
    const inputSnapshot = getFormSnapshot(formData);

    fetch("/predict", {
      method: "POST",
      body: formData
    })
      .then(function (response) {
        if (!response.ok) {
          throw new Error("Prediction request failed");
        }
        return response.json();
      })
      .then(function (data) {
        const nextPrediction = buildPredictionState({
          sleepScore: Number(data.sleep || 0),
          stressScore: Number(data.stress_score || 0),
          stressLevel: String(data.level || "Medium"),
          recommendation: String(data.doctor_suggestion || data.advice || DEFAULT_PREDICTION.recommendation)
        }, inputSnapshot);

        hasPredictionResult = true;
        renderPrediction(nextPrediction);
        activateView("prediction");
        openResultModal();
        appendPredictionToChart(nextPrediction.sleepScore, inputSnapshot.current_stress);
        loadHistory();
      })
      .catch(function () {
        hasPredictionResult = false;
        renderPrediction(buildPredictionState({
          sleepScore: 0,
          stressScore: 0,
          stressLevel: "Unknown",
          recommendation: "Prediction service is temporarily unavailable. Please try again."
        }, inputSnapshot));
        activateView("prediction");
        openResultModal();
      });
  }

  function renderPrediction(payload) {
    currentPrediction = payload;
    const sleepScore = Math.max(0, Math.min(100, Number(payload.sleepScore) || 0));
    const ringAngle = Math.round((sleepScore / 100) * 360);
    if (refs.scoreRing) {
      refs.scoreRing.style.setProperty("--ring-angle", String(ringAngle) + "deg");
    }
    if (refs.sleepScoreValue) {
      refs.sleepScoreValue.textContent = String(sleepScore);
    }
    if (refs.sleepScoreTag) {
      refs.sleepScoreTag.textContent = String(payload.sleepTag || "");
    }
    if (refs.sleepScoreNote) {
      refs.sleepScoreNote.textContent = String(payload.sleepNote || "");
    }
    if (refs.stressScale) {
      refs.stressScale.textContent = String(payload.stressScale || "");
    }
    if (refs.recommendationText) {
      refs.recommendationText.textContent = String(payload.recommendation || "");
    }
    if (refs.meaningText) {
      refs.meaningText.textContent = String(payload.meaningText || "");
    }
    if (refs.stressPill) {
      refs.stressPill.textContent = String(payload.stressLevel || "").toUpperCase();
    }
    if (refs.overallStatusBadge) {
      refs.overallStatusBadge.textContent = String((payload.overallStatus || payload.stressLevel || "Medium") + " STATUS").toUpperCase();
      refs.overallStatusBadge.classList.remove("status-low", "status-medium", "status-high");
      refs.overallStatusBadge.classList.add("status-" + getLevelClass(payload.overallStatus || payload.stressLevel));
    }
    if (refs.overallStatusNote) {
      refs.overallStatusNote.textContent = String(payload.overallNote || "");
    }
    if (refs.breakdownSleepDuration) {
      refs.breakdownSleepDuration.textContent = String(payload.breakdownSleepDuration || "");
    }
    if (refs.breakdownDeepSleep) {
      refs.breakdownDeepSleep.textContent = String(payload.deepSleep || "");
    }
    if (refs.breakdownStressScale) {
      refs.breakdownStressScale.textContent = String(payload.stressScaleShort || payload.stressScale || "");
    }
    if (refs.breakdownEfficiency) {
      refs.breakdownEfficiency.textContent = String(payload.efficiency || "");
    }
    if (refs.outlookValue) {
      refs.outlookValue.textContent = String(payload.outlookValue || "");
    }
    if (refs.outlookMessage) {
      refs.outlookMessage.textContent = String(payload.outlookMessage || "");
    }
    renderRecommendations(payload.recommendations || []);
    renderFactorStates(payload.factors || {});

    const level = String(payload.stressLevel || "Moderate");
    if (refs.stressLevelText) {
      refs.stressLevelText.textContent = level;
      refs.stressLevelText.classList.remove("low", "moderate", "high");
      refs.stressLevelText.classList.add(getLevelClass(level));
    }

    initializeIcons();
    updateResultModal();
  }

  function getFormSnapshot(formData) {
    return {
      age: Number(formData.get("age") || 0),
      sleep_duration: Number(formData.get("sleep_duration") || 0),
      sleep_quality: Number(formData.get("sleep_quality") || 0),
      daily_steps: Number(formData.get("daily_steps") || 0),
      activity: Number(formData.get("activity") || 0),
      resting_hr: Number(formData.get("resting_hr") || 0),
      current_stress: Number(formData.get("current_stress") || 0)
    };
  }

  function buildPredictionState(base, formValues) {
    const sleepScore = Math.max(0, Math.min(100, Number(base.sleepScore) || 0));
    const stressScore = Math.max(0, Math.min(100, Number(base.stressScore) || (Number(formValues.current_stress) || 0) * 10));
    const stressLevel = String(base.stressLevel || "Medium");
    const efficiency = hasPredictionResult
      ? Math.max(55, Math.min(98, Math.round((Number(formValues.sleep_quality) || 0) * 8 + (Number(formValues.sleep_duration) || 0) * 8)))
      : 0;
    const deepSleep = hasPredictionResult
      ? Math.max(12, Math.min(32, Math.round((Number(formValues.sleep_quality) || 0) * 3 + 8)))
      : 0;
    const outlookDelta = hasPredictionResult
      ? Math.max(-18, Math.min(35, Math.round((sleepScore - stressScore) / 4)))
      : 0;
    const recommendations = buildRecommendations(formValues, sleepScore, stressScore);

    return {
      sleepScore: sleepScore,
      stressScore: stressScore,
      stressLevel: stressLevel,
      sleepTag: !hasPredictionResult ? "Pending" : sleepScore >= 75 ? "Good" : sleepScore >= 50 ? "Moderate" : "Needs Care",
      stressScale: buildStressCopy(stressLevel),
      stressScaleShort: buildRemSleepCopy(sleepScore, formValues),
      recommendation: String(base.recommendation || DEFAULT_PREDICTION.recommendation),
      meaningText: buildMeaningText(stressScore),
      overallStatus: deriveOverallStatus(sleepScore, stressScore),
      overallNote: buildOverallNote(sleepScore, stressScore),
      sleepNote: buildSleepNote(sleepScore),
      breakdownSleepDuration: hasPredictionResult ? formatHoursAndMinutes(formValues.sleep_duration) : "0h 00m",
      deepSleep: formatDeepSleep(formValues.sleep_duration, deepSleep),
      efficiency: `${efficiency}%`,
      outlookValue: !hasPredictionResult ? "0%" : outlookDelta >= 0 ? `+${outlookDelta}%` : `${outlookDelta}%`,
      outlookMessage: buildOutlookMessage(outlookDelta),
      recommendations: recommendations,
      factors: {
        heartRate: {
          value: hasPredictionResult ? normalizePercent(formValues.resting_hr, 55, 110) : 0,
          state: hasPredictionResult ? classifyRange(formValues.resting_hr, 70, 85) : "Low",
          meta: `${Number(hasPredictionResult ? formValues.resting_hr : 0)} bpm`
        },
        sleepDuration: {
          value: hasPredictionResult ? normalizePercent(formValues.sleep_duration, 4, 9) : 0,
          state: hasPredictionResult ? classifySleepDuration(formValues.sleep_duration) : "Low",
          meta: hasPredictionResult ? formatHoursAndMinutes(formValues.sleep_duration) : "0h 00m"
        },
        activity: {
          value: hasPredictionResult ? normalizePercent(formValues.activity, 0, 60) : 0,
          state: hasPredictionResult ? classifyInverseRange(formValues.activity, 20, 40) : "Low",
          meta: `${Number(hasPredictionResult ? formValues.daily_steps : 0).toLocaleString()} steps`
        },
        stress: {
          value: hasPredictionResult ? normalizePercent(formValues.current_stress, 0, 10) : 0,
          state: hasPredictionResult ? classifyRange(formValues.current_stress, 4, 7) : "Low",
          meta: hasPredictionResult ? formatScreenTime(formValues.current_stress) : "0h 00m"
        },
        sleepQuality: {
          value: hasPredictionResult ? normalizePercent(formValues.sleep_quality, 1, 10) : 0,
          state: hasPredictionResult ? (formValues.sleep_quality >= 7 ? "Good" : classifyInverseRange(formValues.sleep_quality, 4, 7)) : "Low",
          meta: hasPredictionResult ? "Stable" : "None"
        }
      }
    };
  }

  function deriveOverallStatus(sleepScore, stressScore) {
    if (stressScore >= 70 || sleepScore <= 45) {
      return "High";
    }
    if (stressScore >= 40 || sleepScore <= 70) {
      return "Medium";
    }
    return "Low";
  }

  function buildOverallNote(sleepScore, stressScore) {
    if (!hasPredictionResult) {
      return "No prediction has been generated yet.";
    }
    if (stressScore >= 70) {
      return "High stress despite moderate sleep indicates lifestyle imbalance.";
    }
    if (sleepScore >= 75 && stressScore < 40) {
      return "Balanced recovery and stress signals suggest a stable day ahead.";
    }
    return "Your stress and recovery signals are mixed, so your habits tonight matter.";
  }

  function buildSleepNote(sleepScore) {
    if (!hasPredictionResult) {
      return "No sleep score yet. Your result will appear after prediction.";
    }
    if (sleepScore >= 75) {
      return "You slept well and your recovery pattern looks strong.";
    }
    if (sleepScore >= 50) {
      return "You had a decent sleep. There is room for improvement.";
    }
    return "Your sleep was not enough for full recovery, so tonight should be a reset.";
  }

  function buildOutlookMessage(outlookDelta) {
    if (!hasPredictionResult) {
      return "Next day outlook will appear after prediction.";
    }
    if (outlookDelta >= 15) {
      return "If you follow the suggested recommendations, your stress may improve by 25-35% and sleep quality by 15-20%.";
    }
    if (outlookDelta >= 0) {
      return "A moderate improvement is likely if you protect your sleep window and lower mental load tonight.";
    }
    return "Without better recovery habits tonight, tomorrow may continue to feel imbalanced.";
  }

  function buildRecommendations(formValues, sleepScore, stressScore) {
    if (!hasPredictionResult) {
      return ["Generate a prediction to unlock personalized recommendations."];
    }
    const tips = [];

    if (formValues.current_stress >= 7) {
      tips.push("Use a 10-minute breathing or meditation break before bedtime.");
    }
    if (formValues.sleep_duration < 7) {
      tips.push("Aim for at least 30-60 more minutes of sleep tonight.");
    }
    if (formValues.daily_steps < 6000) {
      tips.push("Add a short walk to lift activity and reduce tension.");
    }
    if (formValues.resting_hr >= 80) {
      tips.push("Keep caffeine lighter later in the day and prioritize hydration.");
    }
    if (formValues.sleep_quality <= 4) {
      tips.push("Create a wind-down routine and dim screens before sleep.");
    }
    if (tips.length < 4 && sleepScore < 70) {
      tips.push("Try a consistent sleep schedule to strengthen overnight recovery.");
    }
    if (tips.length < 5 && stressScore < 50) {
      tips.push("Maintain the habits that are already keeping stress relatively steady.");
    }

    return tips.slice(0, 5);
  }

  function buildStressCopy(stressLevel) {
    if (!hasPredictionResult) {
      return "No stress analysis yet. Run a prediction to see your result.";
    }
    if (String(stressLevel).toLowerCase().indexOf("high") !== -1) {
      return "Your stress level is higher than normal. Take action to reduce stress.";
    }
    if (String(stressLevel).toLowerCase().indexOf("low") !== -1) {
      return "Your stress markers are well-managed today. Keep your current rhythm going.";
    }
    return "Your stress level is slightly elevated. Small recovery habits can improve it.";
  }

  function buildMeaningText(stressScore) {
    if (!hasPredictionResult) {
      return "Your result will appear here after you generate a prediction.";
    }
    if (stressScore >= 70) {
      return "Your stress is affecting your overall well-being. Prioritize relaxation, physical activity and better sleep habits.";
    }
    if (stressScore >= 40) {
      return "Your body is coping, but it still needs steadier sleep and better stress balance.";
    }
    return "Your stress is fairly controlled right now, so maintaining your routine should help.";
  }

  function formatHoursAndMinutes(hours) {
    const safeHours = Number(hours) || 0;
    const wholeHours = Math.floor(safeHours);
    const minutes = Math.round((safeHours - wholeHours) * 60);
    return `${wholeHours}h ${String(minutes).padStart(2, "0")}m`;
  }

  function formatDeepSleep(hours, percent) {
    if (!hasPredictionResult) {
      return "0h 00m (0%)";
    }
    const durationHours = Math.max(0, (Number(hours) || 0) * (Number(percent) || 0) / 100);
    return `${formatHoursAndMinutes(durationHours)} (${percent}%)`;
  }

  function buildRemSleepCopy(sleepScore, formValues) {
    if (!hasPredictionResult) {
      return "0h 00m (0%)";
    }
    const remPercent = Math.max(18, Math.min(30, Math.round((Number(formValues.sleep_quality) || 0) * 2.5 + 8)));
    const remHours = Math.max(0, (Number(formValues.sleep_duration) || 0) * remPercent / 100);
    return `${formatHoursAndMinutes(remHours)} (${remPercent}%)`;
  }

  function formatScreenTime(stressInput) {
    const hours = Math.max(1.2, Math.min(3.8, Number(stressInput || 0) * 0.38));
    return formatHoursAndMinutes(hours);
  }

  function renderRecommendations(items) {
    if (!refs.recommendationsList) {
      return;
    }
    const icons = ["person-standing", "smartphone", "person-standing", "bed-single", "cup-soda"];
    refs.recommendationsList.innerHTML = items.map(function (item, index) {
      const icon = icons[Math.min(index, icons.length - 1)] || "sparkles";
      return '<li><i data-lucide="' + icon + '"></i>' + escapeHtml(item) + "</li>";
    }).join("");
  }

  function renderFactorStates(factors) {
    updateFactor(refs.factorHeartRateState, refs.factorHeartRateBar, factors.heartRate);
    updateFactor(refs.factorSleepDurationState, refs.factorSleepDurationBar, factors.sleepDuration);
    updateFactor(refs.factorActivityState, refs.factorActivityBar, factors.activity);
    updateFactor(refs.factorStressState, refs.factorStressBar, factors.stress);
    updateFactor(refs.factorSleepQualityState, refs.factorSleepQualityBar, factors.sleepQuality);
    setText(refs.factorHeartRateMeta, factors.heartRate && factors.heartRate.meta);
    setText(refs.factorSleepDurationMeta, factors.sleepDuration && factors.sleepDuration.meta);
    setText(refs.factorActivityMeta, factors.activity && factors.activity.meta);
    setText(refs.factorStressMeta, factors.stress && factors.stress.meta);
    setText(refs.factorSleepQualityMeta, factors.sleepQuality && factors.sleepQuality.meta);
  }

  function updateFactor(stateNode, barNode, factor) {
    if (!factor) {
      return;
    }
    if (stateNode) {
      const nextClass = getLevelClass(factor.state);
      stateNode.textContent = factor.state;
      stateNode.classList.remove("low", "moderate", "high", "good");
      stateNode.classList.add(nextClass === "moderate" ? "moderate" : nextClass);
    }
    if (barNode) {
      barNode.style.width = `${factor.value}%`;
    }
  }

  function getLevelClass(level) {
    const normalized = String(level || "").toLowerCase();
    if (normalized.indexOf("low") !== -1) {
      return "low";
    }
    if (normalized.indexOf("high") !== -1) {
      return "high";
    }
    return "moderate";
  }

  function normalizePercent(value, min, max) {
    const safeValue = Number(value) || 0;
    const percent = ((safeValue - min) / (max - min)) * 100;
    return Math.max(10, Math.min(100, Math.round(percent)));
  }

  function classifyRange(value, mediumCutoff, highCutoff) {
    if (value >= highCutoff) {
      return "High";
    }
    if (value >= mediumCutoff) {
      return "Moderate";
    }
    return "Low";
  }

  function classifyInverseRange(value, lowCutoff, mediumCutoff) {
    if (value < lowCutoff) {
      return "High";
    }
    if (value < mediumCutoff) {
      return "Moderate";
    }
    return "Low";
  }

  function classifySleepDuration(value) {
    if (value < 6) {
      return "High";
    }
    if (value < 7) {
      return "Moderate";
    }
    return "Low";
  }

  function escapeHtml(value) {
    const div = document.createElement("div");
    div.textContent = String(value || "");
    return div.innerHTML;
  }

  function setText(node, value) {
    if (node && typeof value !== "undefined") {
      node.textContent = String(value);
    }
  }

  function updateResultModal() {
    if (refs.resultModalSleepScore) {
      refs.resultModalSleepScore.textContent = refs.sleepScoreValue.textContent;
    }
    if (refs.resultModalStressLevel) {
      refs.resultModalStressLevel.textContent = refs.stressLevelText.textContent;
    }
    if (refs.resultModalStressScale) {
      refs.resultModalStressScale.textContent = refs.stressScale.textContent;
    }
    if (refs.resultModalRecommendation) {
      refs.resultModalRecommendation.textContent = refs.recommendationText.textContent;
    }
  }

  function openResultModal() {
    updateResultModal();
    if (refs.resultModal) {
      refs.resultModal.classList.remove("hidden-panel");
      refs.resultModal.setAttribute("aria-hidden", "false");
    }
  }

  function closeResultModal() {
    if (refs.resultModal) {
      refs.resultModal.classList.add("hidden-panel");
      refs.resultModal.setAttribute("aria-hidden", "true");
    }
  }

  function loadHistory() {
    fetch("/history")
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        if (!data.labels || data.labels.length === 0) {
          renderHistoryChart(SAMPLE_HISTORY);
          setHistoryText("Showing sample trend data.", false);
          return;
        }
        renderHistoryChart(data);
        setHistoryText("Trend updated from your latest records.", false);
      })
      .catch(function () {
        renderHistoryChart(SAMPLE_HISTORY);
        setHistoryText("Showing sample trend data.", false);
      });
  }

  function setHistoryText(text, isEmptyState) {
    if (refs.historyState) {
      refs.historyState.textContent = text;
      refs.historyState.classList.toggle("empty-state", Boolean(isEmptyState));
    }
  }

  function renderHistoryChart(data) {
    if (!refs.historyChart) {
      return;
    }

    if (historyChart) {
      historyChart.destroy();
    }

    const textMain = getComputedStyle(refs.body).getPropertyValue("--text-main").trim();
    const textSoft = getComputedStyle(refs.body).getPropertyValue("--text-soft").trim();
    const isLight = refs.body.classList.contains("light-mode");
    const gridColor = isLight ? "rgba(15, 23, 42, 0.16)" : "rgba(148, 163, 184, 0.08)";

    historyChart = new Chart(refs.historyChart, {
      type: "line",
      data: {
        labels: data.labels,
        datasets: [
          {
            label: "Sleep Score",
            data: data.sleep,
            borderColor: "#2dd4bf",
            backgroundColor: "rgba(45, 212, 191, 0.18)",
            pointRadius: 4,
            pointHoverRadius: 5,
            pointBackgroundColor: "#2dd4bf",
            borderWidth: 2.2,
            tension: 0.34
          },
          {
            label: "Stress Level",
            data: data.stress,
            borderColor: "#a78bfa",
            backgroundColor: "rgba(167, 139, 250, 0.16)",
            pointRadius: 4,
            pointHoverRadius: 5,
            pointBackgroundColor: "#a78bfa",
            borderWidth: 2,
            tension: 0.34
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: {
              color: textMain,
              usePointStyle: true,
              pointStyle: "circle",
              boxWidth: 10,
              boxHeight: 10,
              font: {
                size: 12
              }
            }
          }
        },
        scales: {
          x: {
            grid: { color: gridColor },
            ticks: { color: textSoft }
          },
          y: {
            beginAtZero: true,
            grid: { color: gridColor },
            ticks: { color: textSoft }
          }
        }
      }
    });
  }

  function appendPredictionToChart(sleepScore, stressScore) {
    const now = new Date();
    const label = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const safeSleep = Math.max(0, Math.min(100, Number(sleepScore) || 0));
    const safeStress = Math.max(0, Math.min(10, Number(stressScore) || 0));

    if (!historyChart) {
      renderHistoryChart({
        labels: [label],
        sleep: [safeSleep],
        stress: [safeStress]
      });
      setHistoryText("Trend updated from your latest prediction.", false);
      return;
    }

    const labels = historyChart.data.labels || [];
    const sleepData = historyChart.data.datasets[0].data || [];
    const stressData = historyChart.data.datasets[1].data || [];

    labels.push(label);
    sleepData.push(safeSleep);
    stressData.push(safeStress);

    if (labels.length > 20) {
      labels.shift();
      sleepData.shift();
      stressData.shift();
    }

    historyChart.update();
    setHistoryText("Trend updated from your latest prediction.", false);
  }

  function clearHistory() {
    showEmptyHistoryGraph();
  }

  function showEmptyHistoryGraph() {
    if (historyChart) {
      historyChart.destroy();
      historyChart = null;
    }
    if (refs.historyChart) {
      const ctx = refs.historyChart.getContext("2d");
      ctx.clearRect(0, 0, refs.historyChart.width, refs.historyChart.height);
    }
    setHistoryText("Your Health Graph", true);
  }

  function setTheme(nextTheme) {
    theme = nextTheme === "light" ? "light" : "dark";
    refs.body.classList.toggle("light-mode", theme === "light");
    localStorage.setItem(THEME_STORAGE_KEY, theme);

    if (refs.themeToggleSidebarText) {
      refs.themeToggleSidebarText.textContent = theme === "light" ? "Dark" : "Light";
    }

    if (historyChart) {
      renderHistoryChart({
        labels: historyChart.data.labels,
        sleep: historyChart.data.datasets[0].data,
        stress: historyChart.data.datasets[1].data
      });
    }
  }

  function restoreTheme() {
    const storedTheme = localStorage.getItem(THEME_STORAGE_KEY);
    setTheme(storedTheme || "dark");
  }

  function toggleTheme() {
    setTheme(theme === "dark" ? "light" : "dark");
  }

  function toggleVoice(event) {
    const requestedState = event && event.target ? Boolean(event.target.checked) : !voiceEnabled;
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    voiceEnabled = requestedState;
    syncVoiceToggles(voiceEnabled);

    if (voiceEnabled) {
      focusChat();
      showVoiceFeedback("Voice assistant is ON. I am ready to listen.");
      announceVoiceAssistantReady();
      if (!Recognition) {
        // Keep assistant usable with speech output fallback when mic recognition is unavailable.
        const fallbackMessage = "Microphone voice recognition is not available in this browser. You can still use Nidra AI by typing in chat.";
        speakAssistantPrompt(fallbackMessage);
        showVoiceFeedback(fallbackMessage);
        return;
      }
      startVoiceRecognition();
    } else {
      stopVoiceRecognition();
      voicePromptSpoken = false;
    }
  }

  function syncVoiceToggles(enabled) {
    if (refs.voiceToggleSwitch) {
      refs.voiceToggleSwitch.checked = enabled;
    }
    if (refs.voiceToggleSettings) {
      refs.voiceToggleSettings.checked = enabled;
    }
  }

  function startVoiceRecognition() {
    if (voiceListening) {
      return;
    }

    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) {
      return;
    }

    voiceRecognizer = new Recognition();
    voiceRecognizer.lang = "en-US";
    voiceRecognizer.interimResults = false;
    voiceRecognizer.maxAlternatives = 1;
    voiceRecognizer.continuous = false;

    voiceRecognizer.onstart = function () {
      voiceListening = true;
    };

    voiceRecognizer.onresult = function (event) {
      const transcript = event.results[0][0].transcript.trim();
      handleVoiceTranscript(transcript);
    };

    voiceRecognizer.onend = function () {
      voiceListening = false;
      if (voiceEnabled) {
        window.setTimeout(startVoiceRecognition, 260);
      }
    };

    voiceRecognizer.onerror = function (event) {
      voiceListening = false;
      if (event && (event.error === "not-allowed" || event.error === "service-not-allowed")) {
        voiceEnabled = false;
        syncVoiceToggles(false);
        const permissionMessage = "Microphone permission is blocked. Please allow microphone access to use voice assistant.";
        speakAssistantPrompt(permissionMessage);
        showVoiceFeedback(permissionMessage);
      }
    };

    try {
      voiceRecognizer.start();
    } catch (error) {
      voiceListening = false;
    }
  }

  function stopVoiceRecognition() {
    if (voiceRecognizer && voiceListening) {
      voiceRecognizer.stop();
    }
    voiceListening = false;
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
  }

  function handleVoiceTranscript(transcript) {
    if (Date.now() < suppressVoiceInputUntil) {
      return;
    }

    const text = transcript.toLowerCase();
    let updatedField = false;

    if (text.indexOf("predict") !== -1) {
      refs.predictionForm.requestSubmit();
      return;
    }

    if (text.indexOf("chat") !== -1) {
      sendChatMessage(transcript, { speakReply: true, source: "voice" });
      return;
    }

    updatedField = applyVoiceValue(text, /age\s+(\d{1,2})/, "age") || updatedField;
    updatedField = applyVoiceValue(text, /sleep duration\s+(\d+(\.\d+)?)/, "sleep_duration") || updatedField;
    updatedField = applyVoiceValue(text, /(stress|stress level)\s+(\d{1,2})/, "current_stress", 2) || updatedField;
    updatedField = applyVoiceValue(text, /sleep quality\s+(\d{1,2})/, "sleep_quality") || updatedField;
    updatedField = applyVoiceValue(text, /steps\s+(\d{2,6})/, "daily_steps") || updatedField;
    updatedField = applyVoiceValue(text, /(heart rate|resting heart rate)\s+(\d{2,3})/, "resting_hr", 2) || updatedField;

    if (updatedField) {
      showVoiceFeedback("I updated your form from voice input. Say predict when you are ready.");
      return;
    }

    sendChatMessage(transcript, { speakReply: true, source: "voice" });
  }

  function announceVoiceAssistantReady() {
    if (voicePromptSpoken) {
      return;
    }
    voicePromptSpoken = true;
    speakAssistantPrompt("Hello, I am Nidra AI. I am ready to listen. How can I help you?");
  }

  function speakAssistantPrompt(message) {
    if (!("speechSynthesis" in window)) {
      if (message) {
        window.setTimeout(function () {
          window.alert(message);
        }, 0);
      }
      return;
    }
    // Prevent recognizer from catching assistant's own voice output.
    suppressVoiceInputUntil = Date.now() + 4500;
    if (voiceEnabled) {
      stopVoiceRecognition();
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(message);
    utterance.lang = "en-US";
    utterance.rate = 1;
    utterance.pitch = 1;
    utterance.onend = function () {
      if (voiceEnabled) {
        window.setTimeout(startVoiceRecognition, 220);
      }
    };
    window.setTimeout(function () {
      window.speechSynthesis.speak(utterance);
    }, 120);
  }

  function showVoiceFeedback(message) {
    if (!message) {
      return;
    }

    const session = getActiveSession();
    if (session) {
      session.messages.push({
        role: "bot",
        text: String(message),
        createdAt: new Date().toISOString()
      });
      session.updatedAt = new Date().toISOString();
      saveChatStore();
      renderActiveChat();
    }
  }

  function applyVoiceValue(source, pattern, fieldName, groupIndex) {
    const match = source.match(pattern);
    if (!match || !refs.predictionForm || !refs.predictionForm[fieldName]) {
      return false;
    }
    refs.predictionForm[fieldName].value = match[groupIndex || 1];
    return true;
  }

  function loadChatStore() {
    try {
      const raw = localStorage.getItem(CHAT_STORAGE_KEY);
      if (!raw) {
        return;
      }
      const parsed = JSON.parse(raw);
      if (!parsed || !Array.isArray(parsed.sessions)) {
        return;
      }
      chatStore.sessions = parsed.sessions;
      chatStore.activeSessionId = parsed.activeSessionId || null;
      chatStore.sessionCounter = Number(parsed.sessionCounter) || chatStore.sessions.length;
    } catch (error) {
      chatStore = { sessions: [], activeSessionId: null, sessionCounter: 0 };
    }
  }

  function saveChatStore() {
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(chatStore));
  }

  function ensureActiveSession() {
    if (chatStore.sessions.length === 0) {
      createSession();
      return;
    }
    if (!getActiveSession()) {
      chatStore.activeSessionId = chatStore.sessions[0].id;
      saveChatStore();
    }
  }

  function createSession() {
    chatStore.sessionCounter += 1;
    const nowIso = new Date().toISOString();
    const session = {
      id: "session-" + Date.now() + "-" + Math.floor(Math.random() * 1000),
      title: "Session " + chatStore.sessionCounter,
      pinned: false,
      createdAt: nowIso,
      updatedAt: nowIso,
      messages: [
        {
          role: "bot",
          text: "Hello. How can I help you improve your sleep and reduce stress today?",
          createdAt: nowIso
        }
      ]
    };
    chatStore.sessions.unshift(session);
    chatStore.activeSessionId = session.id;
    editingSessionId = null;
    saveChatStore();
  }

  function getActiveSession() {
    return chatStore.sessions.find(function (session) {
      return session.id === chatStore.activeSessionId;
    });
  }

  function setActiveSession(sessionId) {
    chatStore.activeSessionId = sessionId;
    saveChatStore();
    renderSessionList();
    renderActiveChat();
  }

  function deleteSession(sessionId) {
    chatStore.sessions = chatStore.sessions.filter(function (session) {
      return session.id !== sessionId;
    });
    if (chatStore.sessions.length === 0) {
      chatStore.activeSessionId = null;
    } else if (!getActiveSession()) {
      chatStore.activeSessionId = chatStore.sessions[0].id;
    }
    editingSessionId = null;
    saveChatStore();
    renderSessionList();
    renderActiveChat();
  }

  function updateSessionTitleFromMessage(session, text) {
    if (!/^Session\s+\d+$/i.test(session.title)) {
      return;
    }
    const trimmed = text.trim();
    session.title = trimmed.length > 24 ? trimmed.slice(0, 24) + "..." : trimmed;
  }

  function renderSessionList() {
    if (!refs.chatSessionList) {
      return;
    }

    refs.chatSessionList.innerHTML = "";
    const sessions = chatStore.sessions.slice().sort(function (a, b) {
      if (a.pinned !== b.pinned) {
        return a.pinned ? -1 : 1;
      }
      return new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime();
    });

    sessions.forEach(function (session) {
      const item = document.createElement("div");
      item.className = "session-item" + (session.pinned ? " pinned" : "");

      if (editingSessionId === session.id) {
        const renameWrap = document.createElement("div");
        renameWrap.className = "session-rename-wrap";

        const input = document.createElement("input");
        input.type = "text";
        input.className = "session-rename-input";
        input.value = session.title;

        const saveBtn = document.createElement("button");
        saveBtn.type = "button";
        saveBtn.className = "session-action-btn";
        saveBtn.textContent = "Save";
        saveBtn.addEventListener("click", function () {
          const nextTitle = input.value.trim();
          if (nextTitle) {
            session.title = nextTitle;
            session.updatedAt = new Date().toISOString();
            saveChatStore();
          }
          editingSessionId = null;
          renderSessionList();
        });

        const cancelBtn = document.createElement("button");
        cancelBtn.type = "button";
        cancelBtn.className = "session-action-btn";
        cancelBtn.textContent = "Cancel";
        cancelBtn.addEventListener("click", function () {
          editingSessionId = null;
          renderSessionList();
        });

        renameWrap.appendChild(input);
        renameWrap.appendChild(saveBtn);
        renameWrap.appendChild(cancelBtn);
        item.appendChild(renameWrap);
      } else {
        const head = document.createElement("div");
        head.className = "session-head";

        const openBtn = document.createElement("button");
        openBtn.type = "button";
        openBtn.className = "session-open-btn" + (session.id === chatStore.activeSessionId ? " active" : "");
        openBtn.textContent = session.title;
        openBtn.addEventListener("click", function () {
          setActiveSession(session.id);
        });

        const meta = document.createElement("div");
        meta.className = "session-meta";
        meta.textContent = formatTime(session.updatedAt);

        const actions = document.createElement("div");
        actions.className = "session-actions";

        const pinBtn = document.createElement("button");
        pinBtn.type = "button";
        pinBtn.className = "session-action-btn";
        pinBtn.textContent = session.pinned ? "Unpin" : "Pin";
        pinBtn.addEventListener("click", function () {
          session.pinned = !session.pinned;
          session.updatedAt = new Date().toISOString();
          saveChatStore();
          renderSessionList();
        });

        const renameBtn = document.createElement("button");
        renameBtn.type = "button";
        renameBtn.className = "session-action-btn";
        renameBtn.textContent = "Rename";
        renameBtn.addEventListener("click", function () {
          editingSessionId = session.id;
          renderSessionList();
        });

        const deleteBtn = document.createElement("button");
        deleteBtn.type = "button";
        deleteBtn.className = "session-action-btn";
        deleteBtn.textContent = "Delete";
        deleteBtn.addEventListener("click", function () {
          deleteSession(session.id);
        });

        actions.appendChild(pinBtn);
        actions.appendChild(renameBtn);
        actions.appendChild(deleteBtn);

        head.appendChild(openBtn);
        item.appendChild(head);
        item.appendChild(meta);
        item.appendChild(actions);
      }

      refs.chatSessionList.appendChild(item);
    });

    initializeIcons();
  }

  function renderActiveChat() {
    if (!refs.chatMessages) {
      return;
    }
    refs.chatMessages.innerHTML = "";

    const session = getActiveSession();
    if (!session) {
      const emptyState = document.createElement("div");
      emptyState.className = "chat-message bot";
      const bubble = document.createElement("div");
      bubble.className = "chat-bubble";
      bubble.textContent = "No chat session. Click + to start a new chat.";
      emptyState.appendChild(bubble);
      refs.chatMessages.appendChild(emptyState);
      return;
    }

    session.messages.forEach(function (message) {
      const messageWrap = document.createElement("div");
      messageWrap.className = "chat-message " + message.role;

      const time = document.createElement("div");
      time.className = "chat-time";
      time.textContent = formatTime(message.createdAt);

      const bubble = document.createElement("div");
      bubble.className = "chat-bubble";
      bubble.textContent = message.text;

      messageWrap.appendChild(time);
      messageWrap.appendChild(bubble);
      refs.chatMessages.appendChild(messageWrap);
    });

    refs.chatMessages.scrollTop = refs.chatMessages.scrollHeight;
    initializeIcons();
  }

  function handleChatSubmit(event) {
    event.preventDefault();
    if (!refs.chatInput) {
      return;
    }
    const message = refs.chatInput.value.trim();
    if (!message) {
      return;
    }
    refs.chatInput.value = "";
    sendChatMessage(message);
  }

  function sendChatMessage(messageText, options) {
    const settings = options || {};
    let session = getActiveSession();
    if (!session) {
      createSession();
      renderSessionList();
      session = getActiveSession();
      if (!session) {
        return;
      }
    }

    const nowIso = new Date().toISOString();
    session.messages.push({
      role: "user",
      text: messageText,
      createdAt: nowIso
    });
    session.updatedAt = nowIso;
    updateSessionTitleFromMessage(session, messageText);
    saveChatStore();
    renderSessionList();
    renderActiveChat();

    fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: messageText })
    })
      .then(function (response) {
        return response.json();
      })
      .then(function (payload) {
        const replyText = String(payload.reply || "I can help with sleep and stress guidance.");
        session.messages.push({
          role: "bot",
          text: replyText,
          createdAt: new Date().toISOString()
        });
        session.updatedAt = new Date().toISOString();
        saveChatStore();
        renderSessionList();
        renderActiveChat();
        if (settings.speakReply) {
          speakAssistantPrompt(replyText);
        }
      })
      .catch(function () {
        const fallbackReply = "The assistant is temporarily unavailable. Please try again.";
        session.messages.push({
          role: "bot",
          text: fallbackReply,
          createdAt: new Date().toISOString()
        });
        session.updatedAt = new Date().toISOString();
        saveChatStore();
        renderSessionList();
        renderActiveChat();
        if (settings.speakReply) {
          speakAssistantPrompt(fallbackReply);
        }
      });
  }

  function focusChat() {
    if (refs.chatModal) {
      refs.chatModal.classList.remove("hidden-panel");
      refs.chatModal.setAttribute("aria-hidden", "false");
    }
    if (refs.chatInput) {
      window.setTimeout(function () {
        refs.chatInput.focus();
      }, 250);
    }
  }

  function closeChatModal() {
    if (!refs.chatModal) {
      return;
    }
    refs.chatModal.classList.add("hidden-panel");
    refs.chatModal.setAttribute("aria-hidden", "true");
  }

  function togglePinChat() {
    const session = getActiveSession();
    if (!session) {
      return;
    }
    session.pinned = !session.pinned;
    session.updatedAt = new Date().toISOString();
    saveChatStore();
    renderSessionList();
  }

  function beginRenameCurrentChat() {
    const session = getActiveSession();
    if (!session) {
      return;
    }
    editingSessionId = session.id;
    if (refs.chatHistoryPanel) {
      refs.chatHistoryPanel.classList.remove("hidden-panel");
    }
    renderSessionList();
  }

  function deleteCurrentChat() {
    const session = getActiveSession();
    if (!session) {
      return;
    }
    deleteSession(session.id);
  }

  function formatTime(dateIso) {
    const date = new Date(dateIso);
    return date.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      hour12: true
    });
  }

  function togglePanel(panel, closePanel) {
    if (!panel) {
      return;
    }
    panel.classList.toggle("hidden-panel");
    if (closePanel) {
      closePanel.classList.add("hidden-panel");
    }
  }

  function openProfile() {
    if (!refs.profilePanel) {
      return;
    }
    hydrateProfileEditor();
    refs.profilePanel.classList.remove("hidden-panel");
    refs.profilePanel.setAttribute("aria-hidden", "false");
  }

  function closeProfile() {
    if (!refs.profilePanel) {
      return;
    }
    refs.profilePanel.classList.add("hidden-panel");
    refs.profilePanel.setAttribute("aria-hidden", "true");
  }

  function hydrateProfileEditor() {
    if (!refs.profileNameInput || !refs.profileEmailInput) {
      return;
    }
    const fallbackName = refs.sidebarUserName ? refs.sidebarUserName.textContent.trim() : "Guest User";
    const fallbackEmail = refs.sidebarUserEmail ? refs.sidebarUserEmail.textContent.trim() : "guest@nidra.ai";
    refs.profileNameInput.value = refs.body.dataset.userName || fallbackName;
    refs.profileEmailInput.value = refs.body.dataset.userEmail || fallbackEmail;

    const extras = getStoredProfileExtras();
    if (refs.profileAgeInput) {
      refs.profileAgeInput.value = extras.age || "";
    }
    if (refs.profileGenderInput) {
      refs.profileGenderInput.value = extras.gender || "";
    }
    if (refs.profileSleepGoalInput) {
      refs.profileSleepGoalInput.value = extras.sleepGoal || "";
    }
  }

  function saveProfile(event) {
    event.preventDefault();
    const payload = {
      name: refs.profileNameInput.value.trim(),
      email: refs.profileEmailInput.value.trim()
    };
    if (!payload.name || !payload.email) {
      return;
    }

    fetch("/auth/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(function (response) {
        return response.json();
      })
      .then(function () {
        refs.body.dataset.userName = payload.name;
        refs.body.dataset.userEmail = payload.email;
        if (refs.sidebarUserName) {
          refs.sidebarUserName.textContent = payload.name;
        }
        if (refs.sidebarUserEmail) {
          refs.sidebarUserEmail.textContent = payload.email;
        }
        saveProfileExtras();
        closeProfile();
      });
  }

  function downloadReport() {
    if (!window.jspdf || !window.jspdf.jsPDF) {
      window.alert("PDF export is not available right now.");
      return;
    }

    const jsPDF = window.jspdf.jsPDF;
    const doc = new jsPDF({
      orientation: "portrait",
      unit: "pt",
      format: "a4"
    });

    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const margin = 42;
    let y = 48;

    const sleepScore = refs.sleepScoreValue ? refs.sleepScoreValue.textContent : "0";
    const sleepStatus = refs.sleepScoreTag ? refs.sleepScoreTag.textContent : "Pending";
    const stressLevel = refs.stressLevelText ? refs.stressLevelText.textContent : "Low";
    const overallStatus = refs.overallStatusBadge ? refs.overallStatusBadge.textContent : "LOW STATUS";
    const aiInsight = refs.recommendationText ? refs.recommendationText.textContent : "No insight yet.";
    const meaningText = refs.meaningText ? refs.meaningText.textContent : "No result yet.";
    const recommendations = refs.recommendationsList
      ? Array.from(refs.recommendationsList.querySelectorAll("li")).map(function (item) {
          return item.textContent.trim();
        })
      : ["No recommendations yet."];

    doc.setFillColor(9, 20, 38);
    doc.roundedRect(margin, y, pageWidth - margin * 2, 88, 18, 18, "F");
    doc.setTextColor(76, 230, 218);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(24);
    doc.text("Sleep Health & Stress Prediction", margin + 22, y + 34);
    doc.setTextColor(230, 238, 250);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(12);
    doc.text("Clinical-style summary report", margin + 22, y + 56);
    doc.text(`Generated: ${new Date().toLocaleString("en-US")}`, margin + 22, y + 74);
    y += 110;

    function drawMetricCard(title, value, subtitle, x, top, width, accent) {
      doc.setFillColor(245, 248, 252);
      doc.setDrawColor(220, 228, 238);
      doc.roundedRect(x, top, width, 88, 16, 16, "FD");
      doc.setTextColor(accent[0], accent[1], accent[2]);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(13);
      doc.text(title, x + 18, top + 24);
      doc.setTextColor(15, 23, 42);
      doc.setFontSize(24);
      doc.text(value, x + 18, top + 52);
      doc.setFont("helvetica", "normal");
      doc.setFontSize(11);
      doc.setTextColor(71, 85, 105);
      doc.text(subtitle, x + 18, top + 72, { maxWidth: width - 36 });
    }

    const cardGap = 14;
    const cardWidth = (pageWidth - margin * 2 - cardGap * 2) / 3;
    drawMetricCard("Sleep Score", `${sleepScore}/100`, `Status: ${sleepStatus}`, margin, y, cardWidth, [33, 197, 182]);
    drawMetricCard("Stress Level", stressLevel, refs.stressScale ? refs.stressScale.textContent : "", margin + cardWidth + cardGap, y, cardWidth, [255, 92, 75]);
    drawMetricCard("Overall Status", overallStatus, refs.overallStatusNote ? refs.overallStatusNote.textContent : "", margin + (cardWidth + cardGap) * 2, y, cardWidth, [232, 174, 63]);
    y += 108;

    function drawSection(title, bodyLines, accentColor) {
      const lineHeight = 17;
      const blockHeight = 34 + bodyLines.length * lineHeight + 18;
      doc.setFillColor(255, 255, 255);
      doc.setDrawColor(220, 228, 238);
      doc.roundedRect(margin, y, pageWidth - margin * 2, blockHeight, 16, 16, "FD");
      doc.setTextColor(accentColor[0], accentColor[1], accentColor[2]);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(15);
      doc.text(title, margin + 18, y + 24);
      doc.setTextColor(51, 65, 85);
      doc.setFont("helvetica", "normal");
      doc.setFontSize(11.5);
      bodyLines.forEach(function (line, index) {
        doc.text(line, margin + 18, y + 46 + index * lineHeight, {
          maxWidth: pageWidth - margin * 2 - 36
        });
      });
      y += blockHeight + 12;
    }

    drawSection("AI Insight", doc.splitTextToSize(aiInsight, pageWidth - margin * 2 - 36), [45, 212, 191]);
    drawSection("What This Means", doc.splitTextToSize(meaningText, pageWidth - margin * 2 - 36), [99, 102, 241]);
    drawSection("Recommendations", recommendations.map(function (item) {
      return `- ${item}`;
    }), [14, 165, 233]);

    if (y > pageHeight - 120) {
      doc.addPage();
      y = 52;
    }

    doc.setFillColor(9, 20, 38);
    doc.roundedRect(margin, pageHeight - 62, pageWidth - margin * 2, 26, 12, 12, "F");
    doc.setTextColor(210, 221, 235);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.text("Generated by Nidra AI • Sleep Health & Stress Prediction", margin + 16, pageHeight - 45);

    doc.save("sleep-health-stress-report.pdf");
  }

  function getStoredProfileExtras() {
    try {
      const raw = localStorage.getItem(PROFILE_STORAGE_KEY);
      if (!raw) {
        return {};
      }
      const parsed = JSON.parse(raw);
      return parsed && typeof parsed === "object" ? parsed : {};
    } catch (error) {
      return {};
    }
  }

  function saveProfileExtras() {
    const extras = {
      age: refs.profileAgeInput ? refs.profileAgeInput.value.trim() : "",
      gender: refs.profileGenderInput ? refs.profileGenderInput.value : "",
      sleepGoal: refs.profileSleepGoalInput ? refs.profileSleepGoalInput.value.trim() : ""
    };
    localStorage.setItem(PROFILE_STORAGE_KEY, JSON.stringify(extras));
  }

  function openAuthModal(mode) {
    if (!refs.authModal) {
      return;
    }
    authMode = mode === "signup" ? "signup" : "login";
    refs.authTitle.textContent = authMode === "signup" ? "Signup" : "Login";
    refs.authSubmitButton.textContent = authMode === "signup" ? "Create Account" : "Continue";
    refs.authNameInput.value = refs.body.dataset.userName || "";
    refs.authEmailInput.value = refs.body.dataset.userEmail || "";
    if (refs.authPasswordInput) {
      refs.authPasswordInput.value = "";
    }
    if (refs.authNameInput) {
      refs.authNameInput.closest("label").classList.toggle("hidden-panel", authMode !== "signup");
      refs.authNameInput.required = authMode === "signup";
    }
    refs.authModal.classList.remove("hidden-panel");
    refs.authModal.setAttribute("aria-hidden", "false");
  }

  function closeAuthModal() {
    if (!refs.authModal) {
      return;
    }
    refs.authModal.classList.add("hidden-panel");
    refs.authModal.setAttribute("aria-hidden", "true");
  }

  function submitAuthForm(event) {
    event.preventDefault();

    const payload = {
      name: refs.authNameInput.value.trim(),
      email: refs.authEmailInput.value.trim(),
      password: refs.authPasswordInput ? refs.authPasswordInput.value : ""
    };
    if (!payload.email || !payload.password || (authMode === "signup" && !payload.name)) {
      return;
    }

    fetch(authMode === "signup" ? "/auth/signup" : "/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(function (response) {
        if (!response.ok) {
          throw new Error("Authentication failed");
        }
        return response.json();
      })
      .then(function () {
        window.location.reload();
      })
      .catch(function () {
        window.alert("Authentication failed. Please check your details and try again.");
      });
  }

  function logoutUser() {
    fetch("/auth/logout", {
      method: "POST",
      headers: { "Content-Type": "application/json" }
    })
      .then(function () {
        window.location.reload();
      });
  }

  window.openSidebar = openSidebar;
  window.toggleTheme = toggleTheme;
  window.openProfile = openProfile;
  window.toggleVoice = toggleVoice;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
}());
