## **NP Enrichment Demo**


<textarea class="form-control" id="input-doc" name="input-doc" rows="6" cols="80" white-space="pre-wrap"></textarea>

<button type="button" class="btn btn-outline-primary" onclick="change_text()">Predefined Examples</button>

<button id="form-submit" class="btn btn-primary btn-sm" type="button" onclick="call_nte_model(); return null;">Submit</button>

### Output

<div class="form-group" id="out-text"></div>

[comment]: <> (<textarea id="out-text" name="out-text" rows="6" cols="80" white-space="pre-wrap"></textarea>)


<script type="text/javascript">
    function call_nte_model() {
        let text = document.getElementById("input-doc").value;
        let formdata = new FormData();
        formdata.append("text", text);
        
        let requestOptions = {
          method: 'POST',
          body: formdata,
          redirect: 'follow'
        };
        
        fetch("https://nlp.biu.ac.il/~lazary/tne/", requestOptions)
          .then(function(response) {
            return response.text()
          }).then(function(body) {
            let out = document.getElementById("out-text");
            out.innerHTML = body;
          });
	}

    function change_text(new_text) {
        document.getElementById("input-doc").value = new_text;
    }

    examples = ['I entered the house, the windows were open.',
                'Adam\'s father went to meet the teacher at his school.',
                'TNE is an NLU task, which focus on relations between noun phrases (NPs) that can be mediated via prepositions. The dataset contains 5,497 documents, annotated exhaustively with all possible links between the NPs in each document.',
            ];

    function change_text() {
        document.getElementById("input-doc").value = examples[Math.floor(Math.random() * examples.length)];
    }

</script>