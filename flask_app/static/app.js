$(document).ready(function() {
    $('#loan-form').on('submit', function(e) {
        e.preventDefault();
        
        // Get form data
        var formData = {
            annual_income: parseFloat($('#annual_income').val()),
            loan_amount: parseFloat($('#loan_amount').val()),
            loan_term: parseInt($('#loan_term').val()),
            interest_rate: parseFloat($('#interest_rate').val()),
            employment_years: parseFloat($('#employment_years').val()),
            home_ownership: $('#home_ownership').val(),
            loan_purpose: $('#loan_purpose').val()
        };
        
        // Send to API
        $.ajax({
            url: '/predict',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(formData),
            success: function(response) {
                // Display result
                $('#result-container').removeClass('d-none');
                
                // Set risk category
                $('#risk-category')
                    .text(response.risk_category)
                    .removeClass('badge-low badge-medium badge-high')
                    .addClass('badge-' + response.risk_category.toLowerCase());
                
                // Set probability
                $('#probability').text((response.probability * 100).toFixed(2) + '%');
                
                // Set explanation
                var $explanationList = $('#explanation-list');
                $explanationList.empty();
                
                response.explanation.forEach(function(item) {
                    $explanationList.append('<li>' + item + '</li>');
                });
            },
            error: function(xhr) {
                alert('Error: ' + xhr.responseJSON.detail);
            }
        });
    });
});