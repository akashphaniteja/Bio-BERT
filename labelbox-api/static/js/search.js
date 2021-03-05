$(document).ready(function() {
    $('.btn-success').click(function(){
        if($('#comment').val() != ""){
            $.ajax({
                type: "POST",
                url: "/available-abstracts",
                data: JSON.stringify({ 'row_data': $('#comment').val() }),
                contentType: "application/json; charset=utf-8",
                dataType: "json",
                success: function(data){
                    if(Object.keys(data).length > 0){
                        $('#result').removeClass('d-none')
                        $('#no-result').addClass('d-none')
                        $('#uid').text(data.row_uid)
                        $('#external-id').text(data.external_id)
                    }else{
                        $('#result').addClass('d-none')
                        $('#no-result').removeClass('d-none')
                    }
                }
            });
            
        }
        
    });

    $(document).ajaxStart(function(){
        $("#myModal").modal('show')
    });
    $(document).ajaxComplete(function(){
        $("#myModal").modal('hide')
    });
 });