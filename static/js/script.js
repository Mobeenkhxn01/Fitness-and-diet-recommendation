document.addEventListener('DOMContentLoaded', () => {
    const form = document.querySelector('form');

    form.addEventListener('submit', function (e) {
        const gender = document.getElementById('gender').value.trim();
        const weight = document.getElementById('weight').value.trim();
        const height = document.getElementById('height').value.trim();

        // Basic form validation
        if (!gender || !weight || !height) {
            alert('Please fill in all the fields!');
            e.preventDefault();
        }
    });
});
