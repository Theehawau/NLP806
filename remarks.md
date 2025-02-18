# **Diacritization: Issues and Solutions**

## **1. Farasa Sequence-to-Sequence Diacritizer**
### **Issue 1: Incorrect Word Order & Extra Words When Latin Characters Are Present**
- **Problem:** When the input contains Latin characters, the model:
  - Misorders words.
  - Adds new, unintended words.
  
- **Example:**
  - **Input Org:**
    ```
    الحركة النسوية كيفاش بدات و علاش بدات و شنوا المطالب متع النسويات ضيفتي ليوم رانية عطافي feminist a book reviewer an english teacher و لfounder متع طبرقة book club و a point كتبت
    ```
  - **Output:**
    ```
    الْحَرْكَة النّْسُويَة كِيفَاشْ بْدَاتْ وْ عْلَاشْ بْدَاتْ وْ شْنُو الْمْطَالْبْ مْتْعَطّْ النّْسُويَاتْ ضَيّْفْتِي لْيُومْ رَانِيَّة عْطَافِي رَانِيَة ( ( LXم ( عْطَافِي 2 LXم فيطس ، فيطس وْ ْلْفُو : وْ
    ```
  - **Issue Observed:** The model generates extra, unrelated words.

- **Solution Implemented:**
  1. Remove Latin characters from the input.
  2. Apply diacritization.
  3. Reinsert the Latin characters in their correct positions.

- **Example Without Latin Characters (Correct Output):**
  - **Input:**
    ```
    الحركة النسوية كيفاش بدات و علاش بدات و شنوا المطالب متع النسويات ضيفتي ليوم رانية عطافي و ل متع طبرقة و كتبت
    ```
  - **Output:**
    ```
    الْحَرْكَة النّْسُويَة كِيفَاشْ بْدَاتْ وْ عْلَاشْ بْدَاتْ وْ شْنُو الْمْطَالْبْ مْتْعَطّْ النّْسُويَاتْ ضَيّْفْتِي لْيُومْ رَانِيَة عْطَافِي وْ لْ مْتْعَطّْ طَبَرْقَة وْ كْتَبْتْ
    ```

  - **Output with Latin:**
    ```
       الْحَرْكَة النّْسُويَة كِيفَاشْ بْدَاتْ وْ عْلَاشْ بْدَاتْ وْ شْنُو الْمْطَالْبْ مْتْعَطّْ النّْسُويَاتْ ضَيّْفْتِي لْيُومْ رَانِيَة عْطَافِي feminist a book reviewer an english teacher وْ لfounder مْتْعَطّْ طَبَرْقَة book club وْ a point كْتَبْتْ 
    ```
### **Issue 2: API Key Limitation**
- **Problem:** The API key allows only **10 requests**, after which the server returns:
  ```
  Error in connecting to the server: b'"usage limit exceeded"'
  ```
- **Fix:** 
  - still not fixed issue.

---

## **2. CATT Model**
### **Issue 1: Latin Characters Are Removed**
- **Problem:** The model completely **removes Latin characters** from the output.
- **Example:**
  - **Input:**
    ```
    الحركة النسوية كيفاش بدات و علاش بدات و شنوا المطالب متع النسويات ضيفتي ليوم رانية عطافي feminist a book reviewer an english teacher و لfounder متع طبرقة book club و a point كتبت
    ```
  - **Output:**
    ```
    الْحَرَكَةُ النِّسْوِيَّةُ كَيْفَاشٍ بَدَاتٍ وَ عَلَاشٍ بَدَاتٍ وَ شَنُّوا الْمَطَالِبَ مُتَعَ النِّسْوِيَّاتِ ضَيْفَتِي لِيَوْمِ رَانِيَةٍ عَطَافِيٌّ وَ لِ مُتَعِ طَبْرَقَةٍ وَ كَتَبْتُ
    ```
  - **Issue Observed:** All Latin words are missing.

- **Solution Implemented:**
  1. Temporarily remove Latin characters before diacritization.
  2. Apply diacritization.
  3. Reinsert Latin characters in their correct positions.


### **Issue 2: CATT diacritizing dialect as MSA**
- **Problem:** The model diacritize the dialects words as modern standard arabic as showen in the above example.
---

### **Summary of Fixes**
| Model  | Issue | Solution Implemented                                                |
|--------|-------|---------------------------------------------------------------------|
| **Farasa** | Misorders words & adds extra words when Latin characters are present | Temporarily remove Latin characters, diacritize, then reinsert them |
| **Farasa** | API key allows only 10 requests | not fixed yet!                                                      |
| **CATT** | Removes Latin characters entirely | Temporarily remove Latin characters, diacritize, then reinsert them |
| **CATT** | diactirizing dialect as MSA | Manual annotation                                                   |
---