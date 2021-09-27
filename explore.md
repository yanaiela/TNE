## **Explore the TNE Dataset**

### Instructions
* Click an NP on the top-right block to see its NP-relations under "NP Relations" below it.
* Choose one of the splits by clicking the corresponding button below


### Splits

<div class="col-md-4">
    <div class="form-group">
        <button id="train" type="button" class="btn btn-primary btn-sm" onclick="changeSplit('train')">Train</button>
        <button id="dev" type="button" class="btn btn-primary btn-sm" onclick="changeSplit('dev')">Dev</button>
        <button id="test" disabled type="button" class="btn btn-primary btn-sm" onclick="changeSplit('test')">Test</button>
        <button id="test-ood" type="button" class="btn btn-primary btn-sm" onclick="changeSplit('test-ood')">Test-OOD</button>
    </div>
</div>


<iframe id="data_interface" src="https://yanaiela.github.io/TNE/data_interface/test/1491.html" title="TNE Explore" width="100%" height="720" frameBorder="0"></iframe>

<button id="full_screen" type="button" class="btn btn-primary btn-sm" onclick="fullScreen()">Go Full-Screen</button>




<script type="text/javascript">
    function changeLayout(element, val) {
        var el = document.getElementsByClassName(element)[0];
        el.style.maxWidth = val;
    }

    changeLayout('main-content','90rem');

    function changeSplit(new_split) {
        var splits_dic = {
            "train": "https://yanaiela.github.io/TNE/data_interface/train/692.html",
            "dev": "https://yanaiela.github.io/TNE/data_interface/dev/1496.html",
            "test": "https://yanaiela.github.io/TNE/data_interface/test/1491.html",
            "test-ood": "https://yanaiela.github.io/TNE/data_interface/ood/10000.html",
        };
        document.getElementById('data_interface').src = splits_dic[new_split];

        document.getElementById(new_split).disabled = true;
        for (var key in splits_dic) {
            if (key != new_split) {
                document.getElementById(key).disabled = false;
            }
        }

    }

    function fullScreen() {
        var url = document.getElementById('data_interface').src;
        window.location.replace(url);

    }

</script>
