var settings = {};

chrome.runtime.onInstalled.addEventListener(() => {
    chrome.storage.sync.set(settings);
});