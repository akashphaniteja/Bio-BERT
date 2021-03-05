$(document).ready(function() {
   load_tbl_servers();

   $(document).ajaxStart(function(){
        $("#myModal").modal('show')
    });
    $(document).ajaxComplete(function(){
        $("#myModal").modal('hide')
    });
});

var task_id = []
function load_tbl_servers(){
    task_id = []
    return $('#tbl-uploaded-annotations').DataTable( {
        "ajax": {
            "url": "/upload-annotation-list",
            "dataType": "json",
        },
        "destroy": true,
        "columns": [
            { "data": "project" },
            { "data": "date" },
            { "data": "submitted_by" },
            { "data": "description" },
            { "data": "task_id" },
            { "data": "upload_name" },
            { "data": "status" },
            {
                data: null,
                className: "center",
                render: function ( data, type, full, meta ) {
                    task_id.push(full.id)
                    if(full.status == 'RUNNING'){
                        details = '<a href="#" class="tm-product-delete-link btn-details edit-tasks" title="update task">'
                        +'<i class="fa fa-refresh tm-product-delete-icon fa-update-tm update-server"'
                        +' onclick="update_upload('+(task_id.length-1) +')"></i></a>';

                        return details;
                    }else{
                        return '----';
                    }
                    // else{
                    //     details = '<a href="#" class="tm-product-delete-link btn-details edit-tasks" title="update task">'
                    //         +'<i class="fa fa-trash tm-product-delete-icon fa-delete-tm delete-server"'
                    //         +' onclick="delete_upload('+(task_id.length-1)+')"></i>'
                    //     +'</a>'

                    //     return details;
                    // }
                }
            }
        ],
        "pageLength": 10
    });
}


function update_upload(id){
    $.get("/update-status?id="+task_id[id], function(data, status){
        load_tbl_servers()
    });
}

function delete_upload(id){
    $.get("/delete?id="+task_id[id], function(data, status){
        load_tbl_servers()
    });
}

function delete_server(id){
    $('#server_id').val(id)
    $('#mi-modal').modal('show');
}

function process_response(response){
    $("#div-message").fadeTo(2000, 500).slideUp(500, function() {
        $("#div-message").fadeOut(500);
    });
    if(response['status'] == 'error'){
       $('#div-message').html('<div class="form-group alert alert-danger alert-dismissible fade show flashes " id="error-msg" role="alert"><span id="message">' + response['message'] + ' </span><button type="button" class="close" data-dismiss="alert" aria-label="Close"><span aria-hidden="true">&times;</span></button></div>');
    }else{
       $('#div-message').html('<div class="form-group alert alert-info alert-dismissible fade show flashes " id="error-msg" role="alert"><span id="message"> ' + response['message'] + ' </span><button type="button" class="close" data-dismiss="alert" aria-label="Close"><span aria-hidden="true">&times;</span></button></div>');
    }
    $("#message").html(response['message'])
    $("#div-message").show();
}