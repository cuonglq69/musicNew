// Function to show a notice with a given id and text
function showNotice(noticeId, text) {
    const notice = document.getElementById(noticeId);
    notice.innerText = text;
    notice.style.display = 'block';
}

// Function to hide a notice with a given id
function hideNotice(noticeId) {
    const notice = document.getElementById(noticeId);
    notice.style.display = 'none';
}

// Function to generate music
function generateMusic() {
    const generateButton = document.getElementById('generateButton');
    generateButton.disabled = true;

    showNotice('generateNotice', 'Creating Music. Estimated time: 12 minutes');

    // Send a request to the '/generate' endpoint to generate music
    fetch('/generate')
        .then(response => response.text())
        .then(() => {
            const musicPlayer = document.getElementById('musicPlayer');
            const musicSource = 'D:/course%20-%20winter%202023/Applied%20Project/Music_creator/song.mid';
            // Update the source of the music player to the generated song
            musicPlayer.src = musicSource;
            musicPlayer.load(); // Load the new source
            musicPlayer.style.display = 'block'; // Show the music player
            musicPlayer.play(); // Play the generated song

            hideNotice('generateNotice');
            showNotice('generatedNotice', 'Your music is available');

            generateButton.disabled = false;
        });
}

// Add an event listener to the generateButton
const generateButton = document.getElementById('generateButton');
generateButton.addEventListener('click', generateMusic);

// Hide the music player initially
const musicPlayer = document.getElementById('musicPlayer');
musicPlayer.style.display = 'none';