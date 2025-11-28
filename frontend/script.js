/*
 * script.js
 *
 * Этот файл реализует логику визуальной клавиатуры, таблицы соответствия
 * и поле для ввода, работающее по новой раскладке. Объект keyMapping задаёт
 * соответствие между физическими кодами клавиш (например, 'KeyA', 'KeyQ')
 * и символами в новой раскладке. При нажатии клавиши соответствующий
 * символ вставляется в текстовое поле, а визуальная клавиша подсвечивается.
 */

// Определяем соответствие кодов клавиш JavaScript (KeyboardEvent.code) и
// символов в новой раскладке SmartType. Раскладка разделена на 3 ряда,
// каждый содержит 10 позиций (включая клавиши с пунктуацией). Выводите
// символы в нижнем регистре; при необходимости верхний регистр будет
// установлен обработчиком клавиш. В отличие от прежней раскладки, теперь
// семикратные клавиши (Semicolon, Comma, Period, Slash) также используются
// для вывода букв.
const keyMapping = {
  // верхний ряд (10 позиций)
  'KeyQ': 'q',  // Q → q
  'KeyW': 'z',  // W → z
  'KeyE': 'x',  // E → x
  'KeyR': 'j',  // R → j
  'KeyT': ';',  // T → ; (пунктуация)
  'KeyY': '/',  // Y → /
  'KeyU': ',',  // U → ,
  'KeyI': '.',  // I → .
  'KeyO': 'v',  // O → v
  'KeyP': 'b',  // P → b
  // средний (домашний) ряд (10 позиций)
  'KeyA': 'r',  // A → r
  'KeyS': 'e',  // S → e
  'KeyD': 't',  // D → t
  'KeyF': 'n',  // F → n
  'KeyG': 'i',  // G → i
  'KeyH': 'o',  // H → o
  'KeyJ': 'a',  // J → a
  'KeyK': 's',  // K → s
  'KeyL': 'd',  // L → d
  'Semicolon': 'l', // ; → l
  // нижний ряд (10 позиций)
  'KeyZ': 'p',  // Z → p
  'KeyX': 'f',  // X → f
  'KeyC': 'u',  // C → u
  'KeyV': 'h',  // V → h
  'KeyB': 'm',  // B → m
  'KeyN': 'c',  // N → c
  'KeyM': 'y',  // M → y
  'Comma': 'g', // , → g
  'Period': 'k',// . → k
  'Slash': 'w' // / → w
};

// Сопоставление кодов клавиш пунктуации с отображаемым символом родной раскладки
// Используется при построении таблиц соответствия, чтобы корректно выводить
// физический символ для таких кодов, как 'Semicolon', 'Comma', 'Period', 'Slash'.
const physicalNameForCode = {
  Semicolon: ';',
  Comma: ',',
  Period: '.',
  Slash: '/'
};

// Список рядов клавиатуры для построения визуальной раскладки. Используем
// стандартные коды клавиш в том порядке, как они расположены на QWERTY.
// Для новой раскладки SmartType каждая строка содержит 10 клавиш. Вторая и третья
// строки включают клавиши пунктуации.
const keyboardRows = [
  // Верхний ряд (10 клавиш)
  ['KeyQ','KeyW','KeyE','KeyR','KeyT','KeyY','KeyU','KeyI','KeyO','KeyP'],
  // Средний ряд (домашний) включает клавишу Semicolon
  ['KeyA','KeyS','KeyD','KeyF','KeyG','KeyH','KeyJ','KeyK','KeyL','Semicolon'],
  // Нижний ряд включает клавиши Comma, Period и Slash
  ['KeyZ','KeyX','KeyC','KeyV','KeyB','KeyN','KeyM','Comma','Period','Slash']
];

/*
 * Переменные и структуры, используемые для реализации обучающих уроков,
 * подсказок, тепловой карты и расчёта метрик. Вы можете расширить
 * эти данные для поддержки нескольких уроков или динамической генерации
 * текстов для тренировки.
 */

// Список уроков. Каждый урок имеет название и набор слов для тренировки.
// Первый урок ориентирован на домашний ряд, как в примере learn.dvorak.nl.
// Второй урок использует буквы верхнего ряда, третий — нижнего, а четвертый
// объединяет все буквы в смешанную практику. При желании вы можете
// расширить этот список, добавив новые уроки или генерируя слова
// динамически. В UI у пользователя будет возможность выбрать урок.
const lessons = [
  {
    name: 'домашний ряд',
    words: [
      'oaten','oats','tout','tat','sans','hot','hoot',
      'atone','shoot','sane','hats','tooth','shah','eon',
      'hunts','sheet','asst','thous','thou','shunt'
    ]
  },
  {
    name: 'верхний ряд',
    // Слова для верхнего ряда составлены из букв d, l, c, u, m, w, f, g, y, p
    // Большинство — простые английские слова или комбинации для тренировки
    words: [
      'clam','palm','mud','gulp','calf','flap','lady','duly','pug',
      'glad','plum','mug','wavy','camp','plug','clamp','mold','flam','cud'
    ]
  },
  {
    name: 'нижний ряд',
    // Слова для нижнего ряда состоят из букв b, v, k, j, x, q, z
    words: [
      'jazz','vex','box','baz','jab','vax','kaz','bax','jax','qv','qix',
      'zax','jib','vox','viz','kb','xz','jax','bq'
    ]
  },
  {
    name: 'смешанный ряд',
    // Смешанная практика сочетает буквы всех рядов для закрепления навыков
    words: [
      'mix','wave','book','flux','lump','quad','hazy','clown',
      'maze','wolf','jot','back','fun','glow','cap','thumb',
      'jump','dock','max','zen'
    ]
  }
];

// Базовые раскладки. Каждая раскладка представляет отображение кода физической клавиши (например, 'KeyA')
// в букву, которая расположена на этой позиции в выбранной родной раскладке. Используется для вывода
// мини‑лейблов на клавишах, чтобы пользователь видел, какую букву он нажимает на своей привычной
// клавиатуре. Раскладки охватывают QWERTY (стандарт), Dvorak и Colemak. Если для кода нет записи,
// мини‑лейбл не отображается.
const baseLayouts = {
  qwerty: {
    'KeyQ': 'Q','KeyW': 'W','KeyE': 'E','KeyR': 'R','KeyT': 'T','KeyY': 'Y','KeyU': 'U','KeyI': 'I','KeyO': 'O','KeyP': 'P',
    'KeyA': 'A','KeyS': 'S','KeyD': 'D','KeyF': 'F','KeyG': 'G','KeyH': 'H','KeyJ': 'J','KeyK': 'K','KeyL': 'L',
    'KeyZ': 'Z','KeyX': 'X','KeyC': 'C','KeyV': 'V','KeyB': 'B','KeyN': 'N','KeyM': 'M',
    // Дополняем символы пунктуации, которые теперь используются как клавиши
    'Semicolon': ';','Comma': ',','Period': '.','Slash': '/'
  },
  dvorak: {
    'KeyQ': "'",'KeyW': ',', 'KeyE': '.', 'KeyR': 'P', 'KeyT': 'Y', 'KeyY': 'F', 'KeyU': 'G', 'KeyI': 'C', 'KeyO': 'R', 'KeyP': 'L',
    'KeyA': 'A','KeyS': 'O','KeyD': 'E','KeyF': 'U','KeyG': 'I','KeyH': 'D','KeyJ': 'H','KeyK': 'T','KeyL': 'N',
    'KeyZ': ';','KeyX': 'Q','KeyC': 'J','KeyV': 'K','KeyB': 'X','KeyN': 'B','KeyM': 'M',
    // Клавиши пунктуации на базовой раскладке Dvorak отображают те же символы
    'Semicolon': ';','Comma': ',','Period': '.','Slash': '/'
  },
  colemak: {
    'KeyQ': 'Q','KeyW': 'W','KeyE': 'F','KeyR': 'P','KeyT': 'G','KeyY': 'J','KeyU': 'L','KeyI': 'U','KeyO': 'Y','KeyP': ';',
    'KeyA': 'A','KeyS': 'R','KeyD': 'S','KeyF': 'T','KeyG': 'D','KeyH': 'H','KeyJ': 'N','KeyK': 'E','KeyL': 'I',
    'KeyZ': 'Z','KeyX': 'X','KeyC': 'C','KeyV': 'V','KeyB': 'B','KeyN': 'K','KeyM': 'M',
    // Клавиши пунктуации в Colemak отображают оригинальные знаки пунктуации
    'Semicolon': ';','Comma': ',','Period': '.','Slash': '/'
  }
};

// Текущая выбранная базовая раскладка. По умолчанию — QWERTY
let currentBaseLayout = 'qwerty';

// Текущий индекс урока. По умолчанию 0 — домашний ряд.
let currentLesson = 0;

// Список слов для текущего урока. При выборе нового урока эта переменная
// переопределяется соответствующим массивом из lessons[]. Используем let
// вместо const, чтобы можно было изменять ссылку.
let lessonWords = lessons[currentLesson].words;

// Индекс следующего слова в lessonWords, которое предстоит набрать
let lessonIndex = 0;
// Текущее набираемое слово (строчные буквы)
let currentTypedWord = '';
// Счётчик правильно набранных слов
let correctWordCount = 0;
// Время начала урока и интервал таймера для отображения секунд
let startTime = null;
let timerInterval = null;

// Массив кодов клавиш, по которым осуществлялся ввод. Используется для
// подсчёта метрик (домашняя строка, баланс рук, биграммы одним пальцем).
const typedCodes = [];
// Счётчик нажатий каждой клавиши для построения тепловой карты
const usageCount = {};

// Словарь: код клавиши -> индекс ряда (0: верхний, 1: домашний, 2: нижний)
const rowIndexByCode = {};
// Набор кодов клавиш, относящихся к левой руке
const leftHandCodes = new Set([
  'KeyQ','KeyW','KeyE','KeyR','KeyT',
  'KeyA','KeyS','KeyD','KeyF','KeyG',
  'KeyZ','KeyX','KeyC','KeyV','KeyB'
]);
// Заполняем rowIndexByCode, чтобы быстро узнавать принадлежность строки
keyboardRows.forEach((row, idx) => {
  row.forEach(code => { rowIndexByCode[code] = idx; });
});

// Маппинг кодов клавиш к номеру пальца (1–8) для подсчёта биграмм
const fingerIndex = {
  // левая рука
  'KeyQ': 1, 'KeyA': 1, 'KeyZ': 1,
  'KeyW': 2, 'KeyS': 2, 'KeyX': 2,
  'KeyE': 3, 'KeyD': 3, 'KeyC': 3,
  'KeyR': 4, 'KeyF': 4, 'KeyV': 4, 'KeyT': 4, 'KeyG': 4, 'KeyB': 4,
  // правая рука
  'KeyY': 5, 'KeyH': 5, 'KeyN': 5, 'KeyU': 5, 'KeyJ': 5, 'KeyM': 5,
  'KeyI': 6, 'KeyK': 6,
  'KeyO': 7, 'KeyL': 7,
  'KeyP': 8
  // Дополняем индексы пальцев для клавиш пунктуации
  , 'Semicolon': 8 // ; и / набираются мизинцем правой руки
  , 'Comma': 6    // , набирается средним пальцем правой руки
  , 'Period': 7   // . набирается безымянным пальцем правой руки
  , 'Slash': 8    // / набирается мизинцем правой руки
};

// --------------------------
// SVG‑иконки для интерфейса
// --------------------------
// Определяем SVG‑фрагменты для иконок солнца, луны, высокого контраста и клавиатуры.
// Эти иконки основаны на наборах Feather Icons (MIT‑лицензия) и адаптированы
// для текущего проекта. Все они нарисованы линиями, поэтому для изменения
// цвета достаточно менять свойство color на родительском элементе.
const ICONS = {
  sun: `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="5"></circle>
      <line x1="12" y1="1" x2="12" y2="3"></line>
      <line x1="12" y1="21" x2="12" y2="23"></line>
      <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
      <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
      <line x1="1" y1="12" x2="3" y2="12"></line>
      <line x1="21" y1="12" x2="23" y2="12"></line>
      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
      <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
    </svg>`,
  moon: `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
    </svg>`,
  contrast: `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="10"></circle>
      <path d="M12 2v20"></path>
    </svg>`,
  // Иконка для кнопки шпаргалки. Используем help‑circle: окружность с вопросительным
  // знаком, чтобы пользователю было понятнее назначение кнопки. Состоит из
  // кольца и кривой вопросительного знака.
  keyboard: `
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="10"></circle>
      <path d="M9.09 9a3 3 0 1 1 5.82 1c0 1.5-1.5 2-2.5 3" fill="none"></path>
      <line x1="12" y1="17" x2="12.01" y2="17"></line>
    </svg>`
};

// При загрузке DOM строим визуальную клавиатуру и таблицы, инициализируем урок, легкость, шпаргалку
document.addEventListener('DOMContentLoaded', () => {
  // Отдельный элемент для всплывающих подсказок больше не нужен
  buildKeyboard();
  buildMappingTable();
  buildLegendTable();
  // Построив клавиатуру, индексируем её по символу для подсветки целевой буквы
  indexKeysByChar();
  // Инициализируем выпадающий список базовых раскладок и обновляем мини‑лейблы
  initBaseSelect();
  // После выбора базовой раскладки нужно обновить подписи на клавишах
  updateBaseLabels();
  // Заполняем выпадающий список уроков и привязываем обработчик изменения
  populateLessonSelect();
  initLesson();
  initInputHandler();
  initLegendHandlers();

  // Инициализация переключателя темы (светлая/тёмная/высококонтрастная)
  initThemeToggle();
});

/**
 * Глобальная карта: символ новой раскладки (в верхнем регистре) -> DOM‑элемент
 * клавиши. Используется, чтобы находить элемент клавиатуры, который
 * соответствует символу, вставляемому по новой раскладке. Карта
 * переинициализируется каждый раз после перестройки клавиатуры.
 */
let charToKeyEl = {};

/**
 * Глобальный элемент подсказки (tooltip), который отображается
 * при наведении на клавишу или при ошибочном вводе. Он создаётся
 * единожды и переиспользуется для всех подсказок.
 */
// Подсказки при наведении и ошибках убраны. Вместо всплывающих подсказок
// каждая клавиша отображает мини‑лейбл родной раскладки прямо на самой
// клавише. Соответствующие функции для подсказок удалены.

/**
 * Проходит по всем клавишам визуальной клавиатуры и заполняет
 * charToKeyEl. На каждой клавише ожидается текстовый контент с
 * символом новой раскладки (например, 'J'). Этот символ приводится в
 * верхний регистр и используется в качестве ключа.
 */
function indexKeysByChar() {
  charToKeyEl = {};
  document.querySelectorAll('.keyboard .key').forEach(el => {
    let ch = '';
    const mainSpan = el.querySelector('.main-label');
    if (mainSpan) {
      ch = (mainSpan.textContent || '').trim().toUpperCase();
    } else {
      ch = (el.textContent || '').trim().toUpperCase();
    }
    if (ch) charToKeyEl[ch] = el;
  });
}

/**
 * Удаляет все классы подсветки с клавиш визуальной клавиатуры. Этот метод
 * вызывается после отпускания клавиши, при потере фокуса текстового
 * поля или переключении вкладки, чтобы избежать «залипания» подсветки.
 */
function clearKeyHighlights() {
  document.querySelectorAll('.keyboard .key').forEach(el => {
    el.classList.remove('pressed', 'pressed-phys', 'target', 'clear');
  });
}

/**
 * Строит DOM-элементы визуальной клавиатуры. Для каждой клавиши создаётся
 * элемент div с классом "key" и атрибутом data-code, содержащим код клавиши.
 * Надпись на клавише соответствует символу новой раскладки или букве QWERTY,
 * если переназначения нет.
 */
function buildKeyboard() {
  const keyboard = document.getElementById('keyboard');
  keyboardRows.forEach(rowCodes => {
    const rowDiv = document.createElement('div');
    rowDiv.classList.add('row');
    rowCodes.forEach(code => {
      const keyDiv = document.createElement('div');
      keyDiv.classList.add('key');
      keyDiv.dataset.code = code;
      // Отображаем символ в новой раскладке; если его нет, отображаем
      // физическую букву (QWERTY) без изменений
      const mapped = keyMapping[code];
      let label = '';
      if (mapped) {
        label = mapped.toUpperCase();
      } else {
        label = code.replace('Key', '');
      }
      // Формируем основную и вторичную подписи (новый символ и родная физическая клавиша)
      // Если есть переназначение, отображаем новый символ в верхнем регистре и физическую клавишу
      // в качестве мини‑лейбла. Если переназначения нет, показываем только физическую клавишу.
      const mainText = mapped ? mapped.toUpperCase() : code.replace('Key', '');
      // Мини‑лейбл зависит от выбранной базовой раскладки; по умолчанию используется QWERTY
      const subText = mapped ? (baseLayouts[currentBaseLayout] && baseLayouts[currentBaseLayout][code] ? baseLayouts[currentBaseLayout][code] : '') : '';
      // Очищаем текстовое содержимое и добавляем отдельные элементы для надписей
      keyDiv.textContent = '';
      const mainSpan = document.createElement('span');
      mainSpan.classList.add('main-label');
      mainSpan.textContent = mainText;
      keyDiv.appendChild(mainSpan);
      if (subText) {
        const subSpan = document.createElement('span');
        subSpan.classList.add('sub-label');
        subSpan.textContent = subText;
        keyDiv.appendChild(subSpan);
      }
      rowDiv.appendChild(keyDiv);
    });
    keyboard.appendChild(rowDiv);
  });

  // После построения клавиатуры переиндексируем клавиши по символам
  indexKeysByChar();
}

/**
 * Заполняет таблицу соответствия клавиш, используя keyMapping. Каждая строка
 * показывает физическую клавишу (буква QWERTY) и символ, который будет
 * вставляться при нажатии по новой раскладке. Таблица сортируется по
 * физическим клавишам в алфавитном порядке для удобства.
 */
function buildMappingTable() {
  const tableBody = document.querySelector('#mapping-table tbody');
  // Если основной таблицы соответствия нет (новый дизайн не содержит эту таблицу),
  // выходим без действий, чтобы избежать ошибок. Вся информация доступна
  // через модальное окно, которое заполняется отдельно.
  if (!tableBody) return;
  // Создаём список записей {physical: 'A', code: 'KeyA', mapped: 'e'}
  const entries = Object.keys(keyMapping).map(code => {
    // Определяем физический символ: для кода вида 'KeyX' берём символ X,
    // для пунктуации используем соответствующую запись из physicalNameForCode
    let physical;
    if (physicalNameForCode.hasOwnProperty(code)) {
      physical = physicalNameForCode[code];
    } else {
      physical = code.replace('Key', '');
    }
    return {
      code,
      physical,
      mapped: keyMapping[code]
    };
  });
  // Сортируем по физическому символу (букве)
  entries.sort((a, b) => a.physical.localeCompare(b.physical));
  entries.forEach(entry => {
    const tr = document.createElement('tr');
    const tdPhys = document.createElement('td');
    tdPhys.textContent = entry.physical.toUpperCase();
    const tdMapped = document.createElement('td');
    tdMapped.textContent = entry.mapped.toUpperCase();
    tr.appendChild(tdPhys);
    tr.appendChild(tdMapped);
    tableBody.appendChild(tr);
  });
}

/**
 * Создаёт содержимое таблицы в модальном окне соответствия клавиш. Вызывается при
 * инициализации. В отличие от buildMappingTable, не добавляет в основную
 * раскладку, а заполняет модальное окно legend-modal-table.
 */
function buildLegendTable() {
  const modalBody = document.querySelector('#legend-modal-table tbody');
  if (!modalBody) return;
  modalBody.innerHTML = '';
  const entries = Object.keys(keyMapping).map(code => {
    // Определяем физический символ для легенды. Если код относится к
    // пунктуационным клавишам, берём соответствующий символ, иначе
    // удаляем префикс 'Key'.
    let physical;
    if (physicalNameForCode.hasOwnProperty(code)) {
      physical = physicalNameForCode[code];
    } else {
      physical = code.replace('Key','');
    }
    return { code, physical, mapped: keyMapping[code] };
  });
  entries.sort((a,b) => a.physical.localeCompare(b.physical));
  entries.forEach(entry => {
    const tr = document.createElement('tr');
    const tdPhys = document.createElement('td');
    tdPhys.textContent = entry.physical.toUpperCase();
    const tdMapped = document.createElement('td');
    tdMapped.textContent = entry.mapped.toUpperCase();
    tr.appendChild(tdPhys);
    tr.appendChild(tdMapped);
    modalBody.appendChild(tr);
  });
}

/**
 * Отрисовывает строку подсказки слов, обновляя классы для пройденных, текущих
 * и будущих слов.
 */
function renderPromptLine() {
  const promptDiv = document.getElementById('prompt-line');
  if (!promptDiv) return;
  promptDiv.innerHTML = lessonWords.map((word, idx) => {
    if (idx < lessonIndex) {
      return `<span class="done">${word}</span>`;
    } else if (idx === lessonIndex) {
      return `<span class="focus">${word}</span>`;
    } else {
      return `<span class="upcoming">${word}</span>`;
    }
  }).join(' ');
}

/**
 * Обновляет прогресс-бар в зависимости от количества завершённых слов.
 */
function updateProgressBar() {
  const bar = document.getElementById('progress-bar');
  if (!bar) return;
  const progress = lessonWords.length > 0 ? (lessonIndex / lessonWords.length) * 100 : 0;
  bar.style.width = `${progress}%`;
}

/**
 * Обновляет статистику WPM, точности и таймера. Для упрощения точность
 * рассчитывается по количеству правильно набранных слов.
 */
function updateStatsPanel() {
  // Получаем элементы статистики в панели урока (wpm, accuracy)
  const wpmEl = document.getElementById('stats-wpm');
  const accEl = document.getElementById('stats-accuracy');
  const timerEl = document.getElementById('stats-timer');
  // Элементы статистики в карточке боковой панели
  const wpmDisplay = document.getElementById('stats-wpm-display');
  const accDisplay = document.getElementById('stats-accuracy-display');

  // Вычисляем WPM и точность
  let wpm = 0;
  if (startTime && lessonIndex > 0) {
    const elapsedMinutes = (Date.now() - startTime) / 60000;
    wpm = lessonIndex / (elapsedMinutes || 1);
  }
  let acc = 100;
  if (lessonIndex > 0) {
    acc = (correctWordCount / lessonIndex) * 100;
  }
  // Обновляем WPM и точность в панели урока
  if (wpmEl) {
    wpmEl.textContent = `WPM ${Math.round(wpm)}`;
  }
  if (accEl) {
    accEl.textContent = `Точность ${acc.toFixed(0)}%`;
  }
  // Таймер обновляется отдельно через startTimerIfNeeded, здесь его не трогаем

  // Обновляем WPM и точность в боковой карточке
  if (wpmDisplay) {
    wpmDisplay.textContent = `${Math.round(wpm)}`;
  }
  if (accDisplay) {
    accDisplay.textContent = `${acc.toFixed(0)}%`;
  }
}

/**
 * Запускает таймер при первом вводе символа. Таймер обновляет значение
 * каждую секунду.
 */
function startTimerIfNeeded() {
  if (!startTime) {
    startTime = Date.now();
    const timerEl = document.getElementById('stats-timer');
    const timerDisplay = document.getElementById('stats-timer-display');
    timerInterval = setInterval(() => {
      const elapsedMs = Date.now() - startTime;
      const totalSeconds = Math.floor(elapsedMs / 1000);
      const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, '0');
      const seconds = String(totalSeconds % 60).padStart(2, '0');
      const value = `${minutes}:${seconds}`;
      if (timerEl) {
        timerEl.textContent = value;
      }
      if (timerDisplay) {
        timerDisplay.textContent = value;
      }
    }, 1000);
  }
}

/**
 * Увеличивает счётчик нажатия клавиши и обновляет тепловую карту. Вызывается
 * после каждого вставленного символа из нашей раскладки.
 * @param {string} code Код клавиши (например, 'KeyA')
 */
function updateHeatmapFor(code) {
  usageCount[code] = (usageCount[code] || 0) + 1;
  // Определяем максимальное количество нажатий, чтобы нормализовать шкалу
  let max = 0;
  for (const c in usageCount) {
    if (usageCount[c] > max) max = usageCount[c];
  }
  // Обновляем стиль каждой клавиши в зависимости от её относительной частоты
  document.querySelectorAll('.key').forEach(keyEl => {
    const c = keyEl.dataset.code;
    const count = usageCount[c] || 0;
    const ratio = max ? count / max : 0;
    // Записываем значение в CSS‑переменную. Пустой атрибут data-score нужен,
    // чтобы сработал селектор [data-score]
    keyEl.style.setProperty('--score', ratio.toFixed(3));
    if (count > 0) {
      keyEl.setAttribute('data-score', '1');
    }
  });
}

/**
 * Обновляет мини-дашборд метрик (домашняя строка, баланс рук, биграммы, композит).
 */
function updateMetrics() {
  const total = typedCodes.length;
  let home = 0, bottom = 0;
  let left = 0, right = 0;
  for (const code of typedCodes) {
    const row = rowIndexByCode[code];
    if (row === 1) home++;
    else if (row === 2) bottom++;
    if (leftHandCodes.has(code)) left++; else right++;
  }
  const homeRatio = total ? home / total : 0;
  const bottomRatio = total ? bottom / total : 0;
  const leftRatio = total ? left / total : 0;
  const rightRatio = total ? right / total : 0;
  let sameFinger = 0;
  for (let i = 1; i < typedCodes.length; i++) {
    const prev = typedCodes[i - 1];
    const curr = typedCodes[i];
    if (fingerIndex[prev] && fingerIndex[curr] && fingerIndex[prev] === fingerIndex[curr]) {
      sameFinger++;
    }
  }
  const bigramRatio = typedCodes.length > 1 ? sameFinger / (typedCodes.length - 1) : 0;
  const composite = (homeRatio * 0.4) + ((1 - bigramRatio) * 0.3) + ((1 - bottomRatio) * 0.3);
  const homeEl = document.getElementById('m-home');
  const balanceEl = document.getElementById('m-balance');
  const bigramEl = document.getElementById('m-awk');
  const compEl = document.getElementById('m-comp');
  if (homeEl) homeEl.textContent = `${(homeRatio * 100).toFixed(1)}%`;
  if (balanceEl) balanceEl.textContent = `${Math.round(leftRatio * 100)}/${Math.round(rightRatio * 100)}`;
  if (bigramEl) bigramEl.textContent = `${(bigramRatio * 100).toFixed(1)}%`;
  if (compEl) compEl.textContent = composite.toFixed(3);
}

/**
 * Завершает набор текущего слова (по пробелу или Enter) и переходит к
 * следующему. Проверяет правильность набора и обновляет статистику.
 */
function finishCurrentWord() {
  const expected = lessonWords[lessonIndex] || '';
  const typed = currentTypedWord.trim();
  if (typed.length > 0) {
    if (typed === expected) {
      correctWordCount++;
    }
    lessonIndex++;
    currentTypedWord = '';
    renderPromptLine();
    updateProgressBar();
    updateStatsPanel();
    // Если завершили все слова урока — останавливаем таймер
    if (lessonIndex >= lessonWords.length) {
      if (timerInterval) clearInterval(timerInterval);
    }
    // После завершения слова всплывающая подсказка о следующей клавише не требуется
  }
  // Если слово пустое (просто пробел), тем не менее обновляем подсказку
  // При пустом слове подсказку не обновляем
  if (typed.length === 0) {
    // ничего не делаем
  }
}

/**
 * Инициализирует отображение подсказки, прогресса и метрик в начале урока.
 */
function initLesson() {
  renderPromptLine();
  updateProgressBar();
  updateStatsPanel();
  updateMetrics();
  // Всплывающие подсказки обновляются динамически, отдельный блок не используется
}

/**
 * Обработчики для всплывающего окна (шпаргалки клавиш). Позволяет открыть окно
 * по кнопке или клавише '?', закрыть по крестику, нажатию Escape или клику
 * вне окна, а также фильтровать таблицу по вводу.
 */
function initLegendHandlers() {
  const overlay = document.getElementById('legend-overlay');
  const openBtn = document.getElementById('legend-button');
  const closeBtn = document.getElementById('legend-close');
  const searchInput = document.getElementById('legend-search');
  if (!overlay) return;
  // Заменяем текст кнопки на иконку клавиатуры и обновляем подпись
  if (openBtn) {
    openBtn.innerHTML = ICONS.keyboard;
    openBtn.setAttribute('aria-label', 'Показать шпаргалку клавиш');
  }
  // Клик по кнопке — открыть окно
  if (openBtn) {
    openBtn.addEventListener('click', () => {
      overlay.classList.remove('hidden');
      if (searchInput) {
        searchInput.value = '';
        filterLegendRows('');
        searchInput.focus();
      }
    });
  }
  // Клик по крестику — закрыть
  if (closeBtn) {
    closeBtn.addEventListener('click', () => {
      overlay.classList.add('hidden');
    });
  }
  // Закрываем по клику вне контента
  overlay.addEventListener('click', e => {
    if (e.target === overlay) {
      overlay.classList.add('hidden');
    }
  });
  // Фильтрация по вводу
  if (searchInput) {
    searchInput.addEventListener('input', () => {
      const term = searchInput.value.trim().toLowerCase();
      filterLegendRows(term);
    });
  }
  // Горячие клавиши
  document.addEventListener('keydown', e => {
    if (e.key === '?' && !overlay.classList.contains('hidden')) {
      overlay.classList.add('hidden');
      e.preventDefault();
      return;
    }
    if (e.key === '?' && overlay.classList.contains('hidden')) {
      overlay.classList.remove('hidden');
      if (searchInput) {
        searchInput.value = '';
        filterLegendRows('');
        searchInput.focus();
      }
      e.preventDefault();
      return;
    }
    if (e.key === 'Escape' && !overlay.classList.contains('hidden')) {
      overlay.classList.add('hidden');
      e.preventDefault();
    }
  });
}

/**
 * Создаёт и заполняет выпадающий список уроков. На каждой опции
 * отображается номер урока и его название (например, «Урок 2 — верхний ряд»).
 * При выборе другого урока сбрасываются все счётчики и состояние,
 * переопределяется массив lessonWords, обновляется подпись и отображение
 * подсказки/прогресса/метрик.
 */
function populateLessonSelect() {
  const select = document.getElementById('lesson-select');
  if (!select) return;
  // Очищаем существующие опции
  select.innerHTML = '';
  lessons.forEach((lesson, idx) => {
    const opt = document.createElement('option');
    opt.value = String(idx);
    opt.textContent = `Урок ${idx + 1} — ${lesson.name}`;
    select.appendChild(opt);
  });
  // Устанавливаем текущий выбранный урок
  select.value = String(currentLesson);
  // Обработчик изменения выбранного урока
  select.addEventListener('change', () => {
    const newIndex = parseInt(select.value, 10);
    if (!isNaN(newIndex) && newIndex >= 0 && newIndex < lessons.length) {
      currentLesson = newIndex;
      lessonWords = lessons[currentLesson].words;
      // Сбрасываем состояние урока
      lessonIndex = 0;
      currentTypedWord = '';
      correctWordCount = 0;
      startTime = null;
      if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
      }
      typedCodes.length = 0;
      // Сбрасываем счётчик использования клавиш
      for (const code in usageCount) {
        delete usageCount[code];
      }
      // Обновляем текст заголовка урока
      const lessonInfo = document.getElementById('lesson-info');
      if (lessonInfo) {
        lessonInfo.textContent = `Урок ${currentLesson + 1} — ${lessons[currentLesson].name}`;
      }
      // Перерисовываем прогресс, статистику и метрики
      renderPromptLine();
      updateProgressBar();
      updateStatsPanel();
      updateMetrics();
      // Всплывающая подсказка будет создана динамически при наведении или ошибке
    }
  });
  // Обновляем подпись урока при первой инициализации
  const lessonInfo = document.getElementById('lesson-info');
  if (lessonInfo) {
    lessonInfo.textContent = `Урок ${currentLesson + 1} — ${lessons[currentLesson].name}`;
  }
  // Отдельного блока подсказки нет, поэтому ничего не обновляем
}

/**
 * Инициализирует выпадающий список базовых (родных) раскладок. Этот список
 * позволяет выбрать, какая физическая раскладка используется у пользователя
 * (QWERTY, Dvorak или Colemak). Выбор влияет на мини‑лейблы на клавишах,
 * которые показывают, какую клавишу пользователь нажимает на своей родной
 * раскладке. При изменении выбора обновляет глобальную переменную
 * currentBaseLayout и вызывает обновление всех мини‑лейблов.
 */
function initBaseSelect() {
  const baseSelect = document.getElementById('base-select');
  if (!baseSelect) return;
  // Устанавливаем текущее значение по умолчанию
  baseSelect.value = currentBaseLayout;
  // Обработчик изменения выбранной раскладки
  baseSelect.addEventListener('change', () => {
    const newLayout = baseSelect.value;
    if (newLayout && baseLayouts[newLayout]) {
      currentBaseLayout = newLayout;
      updateBaseLabels();
    }
  });
}

/**
 * Обновляет текст мини‑лейблов (sub-label) на каждой клавише визуальной клавиатуры
 * согласно выбранной базовой раскладке. Проходит по всем клавишам, находит
 * символ родной раскладки для данного кода и обновляет или создаёт
 * соответствующий элемент .sub-label. Если для клавиши нет соответствия
 * в текущей базовой раскладке, мини‑лейбл удаляется.
 */
function updateBaseLabels() {
  const keys = document.querySelectorAll('.keyboard .key');
  keys.forEach(keyEl => {
    const code = keyEl.dataset.code;
    // Находим символ родной раскладки для данного кода
    const baseChar = baseLayouts[currentBaseLayout] && baseLayouts[currentBaseLayout][code] ? baseLayouts[currentBaseLayout][code] : '';
    // Ищем существующий суб‑лейбл
    let subSpan = keyEl.querySelector('.sub-label');
    if (baseChar) {
      if (!subSpan) {
        // Создаём новый суб‑лейбл, если его нет
        subSpan = document.createElement('span');
        subSpan.classList.add('sub-label');
        keyEl.appendChild(subSpan);
      }
      subSpan.textContent = baseChar;
    } else {
      // Если символа нет, удаляем существующий суб‑лейбл
      if (subSpan) {
        subSpan.remove();
      }
    }
  });
  // Переиндексируем клавиши по символам после обновления надписей
  indexKeysByChar();
}

/**
 * Фильтрует строки в модальной таблице, скрывая те, которые не содержат
 * поисковый запрос.
 * @param {string} term поисковый запрос в нижнем регистре
 */
function filterLegendRows(term) {
  const rows = document.querySelectorAll('#legend-modal-table tbody tr');
  rows.forEach(row => {
    const phys = row.children[0].textContent.toLowerCase();
    const mapped = row.children[1].textContent.toLowerCase();
    if (!term || phys.includes(term) || mapped.includes(term)) {
      row.style.display = '';
    } else {
      row.style.display = 'none';
    }
  });
}

/**
 * Обновляет подсказку над клавиатурой, показывая, какую физическую клавишу
 * нужно нажать на родной раскладке, чтобы получить следующий символ новой раскладки.
 * Выводит подсказку в элемент с id="next-hint" в формате:
 * «Сейчас нажмите <phys> → <new>».
 * Если урок завершён, показывает сообщение о завершении.
 */
// Функция обновления подсказки была удалена, поскольку отдельный блок
// подсказок больше не используется. Всплывающие подсказки отображаются
// динамически при наведении и при ошибках.

/**
 * Инициализирует обработчики события для textarea, чтобы перехватывать
 * нажатия клавиш и вставлять соответствующий символ из новой раскладки.
 */
function initInputHandler() {
  const textarea = document.getElementById('input');
  // Подсветка физической и целевой клавиши при нажатии. Вызывается на уровне
  // документа, чтобы реагировать, даже если фокус не на textarea. Игнорируем
  // системные клавиши (Ctrl, Meta, Alt).
  // Глобальный обработчик подсветки: при каждом нажатии физической клавиши
  // кратковременно подсвечиваем соответствующий элемент визуальной клавиатуры.
  // Системные комбинации (Ctrl, Meta, Alt) игнорируются. Подсветка
  // держится около секунды и затем автоматически снимается.
  document.addEventListener('keydown', event => {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    const physCode = event.code;
    // Найти DOM‑элемент клавиши и добавить класс для зелёной подсветки
    const physEl = document.querySelector(`.keyboard .key[data-code="${physCode}"]`);
    if (physEl) {
      physEl.classList.add('pressed');
      // Снимаем подсветку через 1 секунду
      setTimeout(() => {
        physEl.classList.remove('pressed');
      }, 1000);
    }
  });

  // Мы больше не используем старую систему подсветки (pressed-phys, target) и
  // тепловую карту, поэтому специальные обработчики keyup/blur удалены.

  // Основной обработчик для поля ввода: переназначение клавиш и обработка пробелов/Enter
  if (textarea) {
    textarea.addEventListener('keydown', event => {
      // Игнорируем системные комбинации клавиш
      if (event.ctrlKey || event.metaKey || event.altKey) {
        return;
      }
      const code = event.code;
      // Если нажали пробел или Enter, завершаем текущее слово
      if (code === 'Space' || code === 'Enter') {
        // Для Enter — вставляем перенос строки вручную и не пропускаем действие
        if (code === 'Enter') {
          event.preventDefault();
          insertAtCaret(textarea, '\n');
        }
        finishCurrentWord();
        // После завершения слова всплывающая подсказка об обновлении не требуется
        return;
      }
      // Проверяем, нужно ли переназначить букву
      const mapped = keyMapping[code];
      if (mapped) {
        event.preventDefault();
        startTimerIfNeeded();
        // Определяем, требуется ли верхний регистр в зависимости от Shift и CapsLock
        const shift = event.getModifierState('Shift');
        const caps = event.getModifierState('CapsLock');
        let output = mapped;
        if (shift ^ caps) {
          output = output.toUpperCase();
        }
        insertAtCaret(textarea, output);
        // Добавляем в текущий набираемый текст (в нижнем регистре)
        currentTypedWord += output.toLowerCase();
        // Обновляем статистику (тепловая карта больше не используется)
        typedCodes.push(code);
        updateMetrics();
        // Больше не сравниваем с ожидаемым символом и не выводим подсказку.
        // Подсказки и подсветка ошибок удалены.
        return;
      }
      // Иначе символ не в нашей раскладке — позволяем стандартное поведение
      // Подсказки обновляются только при наведении и ошибках, ничего не делаем
    });
  }
}

/**
 * Вставляет заданный текст в текущую позицию курсора текстового поля
 * и перемещает курсор в конец вставки.
 * @param {HTMLTextAreaElement|HTMLInputElement} el - элемент ввода
 * @param {string} text - текст для вставки
 */
function insertAtCaret(el, text) {
  const start = el.selectionStart;
  const end = el.selectionEnd;
  const value = el.value;
  el.value = value.slice(0, start) + text + value.slice(end);
  const newPos = start + text.length;
  el.selectionStart = el.selectionEnd = newPos;
}

/**
 * Инициализирует переключатель темы. Загружает сохранённое значение из
 * localStorage и обновляет иконку. При клике переключает класс 'dark' на body
 * и сохраняет выбранную тему.
 */
function initThemeToggle() {
  const body = document.body;
  const toggle = document.getElementById('theme-toggle');
  if (!toggle) return;

  /**
   * Обновляет содержимое иконки и подпись переключателя в соответствии с
   * текущим состоянием темы. Кнопка отображает иконку следующего состояния,
   * чтобы пользователь понимал, что произойдет при клике.
   */
  function refreshToggle() {
    if (body.classList.contains('high-contrast')) {
      // Сейчас активен высококонтрастный режим – следующий будет светлый
      toggle.innerHTML = ICONS.sun;
      toggle.setAttribute('aria-label', 'Переключить на светлый режим');
    } else if (body.classList.contains('dark')) {
      // Сейчас активен тёмный режим – следующий будет высококонтрастный
      toggle.innerHTML = ICONS.contrast;
      toggle.setAttribute('aria-label', 'Переключить на высококонтрастный режим');
    } else {
      // Сейчас активен светлый режим – следующий будет тёмный
      toggle.innerHTML = ICONS.moon;
      toggle.setAttribute('aria-label', 'Переключить на тёмный режим');
    }
  }

  // Применяем сохранённую тему при загрузке страницы
  const saved = localStorage.getItem('theme');
  if (saved === 'dark') {
    body.classList.add('dark');
  } else if (saved === 'high-contrast') {
    body.classList.add('high-contrast');
  }
  refreshToggle();

  // Обработчик клика по переключателю. Циклически меняет режимы: светлый → тёмный →
  // высококонтрастный → светлый.
  toggle.addEventListener('click', () => {
    if (body.classList.contains('dark')) {
      // Тёмный → высококонтрастный
      body.classList.remove('dark');
      body.classList.add('high-contrast');
      localStorage.setItem('theme', 'high-contrast');
    } else if (body.classList.contains('high-contrast')) {
      // Высококонтрастный → светлый
      body.classList.remove('high-contrast');
      localStorage.setItem('theme', 'light');
    } else {
      // Светлый → тёмный
      body.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    }
    refreshToggle();
  });
}