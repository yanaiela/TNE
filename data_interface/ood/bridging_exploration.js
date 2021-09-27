//-------------- Functions to navigate the nps --------------
      
      
      //This function returns the np that is currently being updated
      function get_current_np() {
      var current_np = nps[current_np_id];
      return current_np;
      }
      
      //This function returns the np that is currently being updated
      function get_current_np_id() {
      var current = current_np_id;
      return current;
      }
      
      //This function changes the value of current_np_id
      //It is executed when the user finishes updating an np and goes to a new one
      function set_current_np(num) {
      current_np_id = num;
      return current_np_id;
      }
	  
      // this function renders the HTML of the main textbox
      function f_render_text(nps, text, entities, cid, bridge_candidates) {
          // this renders the text based on the nps, the text, and the current np id.					 
          var chars = text.raw_text.split('');
          					 
          nps.forEach(np => {
              var my_entity = entities.filter(entity => entity.members.includes(np.id))[0];
              
              if (np.id == cid) {
                  chars[np.start_index] = `<span id='cur_NP'>` + chars[np.start_index];
              } else {
                  if (my_entity != undefined) {                        
                          chars[np.start_index] = "<span class = 'entity" + my_entity.id + "'>" + chars[np.start_index];                        
                  } else {
                      chars[np.start_index] = `<span>` + chars[np.start_index];
                  }
              }
              chars[np.end_index - 1] = chars[np.end_index - 1] + "</span>";
          });
          //title
      if (text.title.start_index < text.title.end_index){	//if there is a title
           chars[text.title.start_index] = `<h2 id = "title">` + chars[text.title.start_index];
           chars[text.title.end_index - 1] = chars[text.title.end_index - 1] + "</h2>";
      }
           //subtitles
      
           text.subtitles.forEach(subtitle => {
      if (subtitle.start_index < subtitle.end_index){//if there is a subtitle
               chars[subtitle.start_index] = `<h3>` + chars[subtitle.start_index];
               chars[subtitle.end_index - 1] = chars[subtitle.end_index - 1] + "</h3>";
      }
           });
          //paragraphs
      <!-- p_counter = 0 -->
           text.paragraphs.forEach((paragraph, index) => {
      if (!'indents' in text){
               chars[paragraph.start_index] = `<p>` + chars[paragraph.start_index];
      } else if ('indents' in text){
      chars[paragraph.start_index] = '<p style="margin-left: ' + text.indents[index]*20 +'px"><b>' + chars[paragraph.start_index]; 
      chars[paragraph.start_index + text.paragraph_authors[index].length] = chars[paragraph.start_index + text.paragraph_authors[index].length] + '</b>';
      }
               chars[paragraph.end_index - 1] = chars[paragraph.end_index - 1] + "</p>";
      <!-- p_counter = p_counter+1 -->
           });
           
           return chars.join("")
       };
       
      
      //this function returns the text of an np
      function f_np_text_from_id(np_id) {
      
          np = nps.filter(np => np.id == np_id)[0];
          return np.text;
      };
      
      
      //this function scrollselements into view
      function scrollElementIntoView(elementId){
      document.getElementById(elementId).scrollIntoView({
          behavior: 'auto',
          block: 'center',
          inline: 'center'
      });		
      }   
      
      
      
      //This function is executed when an item is clicked on the item pane.
      //The clicked item is highlighted in the text. If the clicked item was "done" (scratched out), it becomes "not done" again.
      function go_to_np(np_id) {
          var new_current = set_current_np(np_id);
          
          _text.set({
              'current': new_current
          });
      
      _entity_pane.set({
              'current': new_current
          });
      
      _item_pane.set({                
      'current_np_id': new_current,
          });			
          
         			
      scrollElementIntoView('cur_NP');
      }      
     
	  
     //This function "folds" the links
      function get_bridging_results(){
      var bridging_results = [];
      
      var folded_res = [];
      BRIDGING_RESULTS_UNFOLDED.forEach(function(lnk){delete lnk["entity_source"];
      lnk["complement"] = lnk ["coref_cluster"];
      delete lnk["coref_cluster"];
      folded_res.push(lnk)});
      bridging_results = _.uniqWith(folded_res, _.isEqual);	
      
      console.log(bridging_results);
      
         BRIDGING_RESULTS = bridging_results;		
      _entity_pane.set({                              
      
      'bridging_results':BRIDGING_RESULTS,
             });     
      }
      
      
         //this function highlights all the elements that belong to the same class
         function highlightClass(cls, color) {
             var elements = document.getElementsByClassName(cls);
             for (var j = 0; j < elements.length; j++) {
                 elements[j].style.backgroundColor = color;
             }
         }