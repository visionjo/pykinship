import copy


def tolist(x):
    """Convert a python object to a singleton list if not already a list"""
    if type(x) is list:
        return x
    else:
        return [x]


class GalleryTemplate(object):
    def __init__(self, media=None):
        self._media = (
            tolist(media) if media is not None else []
        )  # list of images or videos
        self._category = tolist(media)[0].category() if media is not None else None
        self._is_flattened = False
        self._encoding = None
        self._templateid = None

    def __repr__(self):
        str_size = ", images=%d, videos=%d, frames=%d" % (
            len(self.images()),
            len(self.videos()),
            len(self.frames()),
        )
        str_category = "category='%s'" % self.category()
        return str("<janus.template: %s%s>" % (str_category, str_size))

    def __len__(self):
        """Total number of images+frames in template"""
        return len(self.images()) + len(self.frames())

    def __iter__(self):
        for m in self._media:
            yield m  # video or image

    def __getitem__(self, k):
        if k >= 0 and k < len(self._media) and len(self._media) > 0:
            return self._media[k]
        else:
            raise ValueError("Invalid media index %d " % k)

    def category(self, newcategory=None):
        if newcategory is None:
            return self._category
        else:
            self._category = newcategory
            return self

    def templateid(self, newid=None):
        """Template ID is the (common) TEMPLATE_ID attribute of all media in template"""
        if newid is not None:
            self._templateid = newid
            return self
        elif self._templateid is not None:
            return self._templateid
        elif (
            (self._media is not None)
            and (len(self._media) > 0)
            and (self._media[0].attributes is not None)
            and ("TEMPLATE_ID" in self._media[0].attributes)
        ):
            return self._media[0].attributes["TEMPLATE_ID"]
        else:
            return None

    def subjectid(self):
        if (
            (self._media is not None)
            and (len(self._media) > 0)
            and (self._media[0].attributes is not None)
            and ("SUBJECT_ID" in self._media[0].attributes)
        ):
            return self._media[0].attributes["SUBJECT_ID"]
        else:
            return None

    def show(self, figure=1, flush=False):
        for m in self:
            if flush:
                m.show(
                    figure=figure
                ).flush()  # flush after display to avoid memory accumulation
            else:
                m.show(figure=figure)  # do not flush after display if no filename

    def images_and_frames(self):
        return self.images() + self.frames()

    def explode(self):
        return [GalleryTemplate(media=[im]) for im in self.images_and_frames()]

    def mediatype(self):
        return "template"

    def media(self):
        return self._media

    def videos(self):
        """List of videos in template"""
        return [m for m in self._media if m.mediatype() == "video"]

    def frames(self):
        """List of video frames in template"""
        return [fr for v in self.videos() for fr in v.frames()]

    def images(self):
        """List of images in template"""
        return [m for m in self._media if m.mediatype() == "image"]

    def flush(self):
        self._media = [m.flush() for m in self._media]
        return self

    def clone(self):
        """Required for maintaining idemponence with certain template transformations"""
        return copy.deepcopy(self)

    def augment(self, m):
        self._category = m.category()
        self._media.append(m)
        return self

    def flatten(self, encoding):
        self._encoding = encoding(self)
        self._media = []
        self._is_flattened = True
        return self

    def enroll(self, category):
        self._category = category
        for m in self._media:
            m = m.category(category)
            m.attributes["TEMPLATE_ID"] = category

    def filter(self, f, preserve=False):
        """Apply function f to filter elements in media list"""
        self._media = [m for m in self.images() if f(m) == True] + [
            m.filter(f) for m in self.videos()
        ]
        if preserve:
            # Don't return an empty template
            return self.clone()
        else:
            return self
