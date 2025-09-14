function createStars() {
    const starsContainer = document.getElementById('stars');
    if (!starsContainer) return;
    const numberOfStars = 100;
    
    for (let i = 0; i < numberOfStars; i++) {
        const star = document.createElement('div');
        star.className = 'star';
        
        star.style.left = Math.random() * 100 + '%';
        star.style.top = Math.random() * 100 + '%';
        
        const size = Math.random() * 3 + 1;
        star.style.width = size + 'px';
        star.style.height = size + 'px';
        
        star.style.animationDelay = Math.random() * 3 + 's';
        
        starsContainer.appendChild(star);
    }
}

document.getElementById('newsForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const headline = document.getElementById('headline').value;
    const country = document.getElementById('country').value;
    const datetime = document.getElementById('datetime').value;
    const resultSection = document.getElementById('resultSection');
    const resultText = document.getElementById('resultText');
    
    // API call adding
    resultText.innerHTML = `
        <strong style="font-size: 1.2rem; color: #a78bfa;">Analysis Complete!</strong><br><br>
        <strong>Headline:</strong> ${headline.substring(0, 100)}${headline.length > 100 ? '...' : ''}<br>
        <strong>Country:</strong> ${country.toUpperCase()}<br>
        <strong>Date:</strong> ${new Date(datetime).toLocaleString()}<br><br>
        <strong>Credibility Score:</strong> ${Math.floor(Math.random() * 40) + 60}%<br>
        <strong>Status:</strong> <span style="color: #34d399; font-weight: bold;">Likely Authentic</span>
    `;
    
    resultSection.classList.add('show');
});

// currTime Default
document.addEventListener('DOMContentLoaded', function() {
    createStars();
    
    const now = new Date();
    const offset = now.getTimezoneOffset();
    const adjustedDate = new Date(now.getTime() - offset * 60 * 1000);
    const datetimeInput = document.getElementById('datetime');
    if(datetimeInput) {
        datetimeInput.value = adjustedDate.toISOString().slice(0, 16);
    }
});
